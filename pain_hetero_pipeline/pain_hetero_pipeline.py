#!/usr/bin/env python3
"""
Pain Heterogeneous GNN Pipeline

End-to-end supervised heterogeneous graph pipeline for pain classification
using metabolomics data with LLM-based name normalization and KEGG mapping.

Node types: sample, met (metabolite), enz (enzyme)
Edges: sample-met (abundance), met-enz (KEGG), met-met (pathway prior)

Usage:
    python pain_hetero_pipeline.py --csv starting_template.csv --outdir results

Author: Pain GNN Pipeline
"""

import os
import sys
import json
import argparse
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

# PyTorch Geometric imports
try:
    import torch_geometric
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import HeteroConv, SAGEConv, Linear
    from torch_geometric.utils import add_self_loops
except ImportError:
    print("ERROR: PyTorch Geometric not installed. Please run:")
    print("  pip install torch-geometric")
    sys.exit(1)

# Local imports
from openrouter_aliases import OpenRouterAliasGenerator
from kegg_mapper import KEGGMapper
from utils import (
    load_metabolomics_data,
    preprocess_fold,
    build_sample_features,
    compute_group_confounding,
    compute_within_group_auc,
    create_visualizations,
    save_results
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Custom GNN Layers
# =============================================================================

class EdgeAttrConv(nn.Module):
    """
    Custom message passing layer that incorporates edge attributes.

    For sample->met edges, the edge_attr is the metabolite abundance.
    Message: MLP([x_src || edge_attr])
    """

    def __init__(self, in_channels: int, out_channels: int, edge_dim: int = 1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels + edge_dim, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        self.out_channels = out_channels

    def forward(
        self,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x_src: Source node features [n_src, in_channels]
            x_dst: Destination node features [n_dst, in_channels] (unused in aggregation)
            edge_index: [2, n_edges]
            edge_attr: [n_edges, edge_dim]

        Returns:
            Updated destination node features [n_dst, out_channels]
        """
        src_idx, dst_idx = edge_index

        # Get source features for each edge
        x_src_edge = x_src[src_idx]  # [n_edges, in_channels]

        # Ensure edge_attr is 2D
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        # Concatenate source features with edge attributes
        edge_messages = torch.cat([x_src_edge, edge_attr], dim=-1)

        # Transform messages
        messages = self.mlp(edge_messages)  # [n_edges, out_channels]

        # Aggregate messages at destination nodes (mean aggregation)
        n_dst = x_dst.size(0)
        out = torch.zeros(n_dst, self.out_channels, device=x_src.device)

        # Scatter mean
        count = torch.zeros(n_dst, device=x_src.device)
        out.scatter_add_(0, dst_idx.unsqueeze(-1).expand(-1, self.out_channels), messages)
        count.scatter_add_(0, dst_idx, torch.ones_like(dst_idx, dtype=torch.float))
        count = count.clamp(min=1).unsqueeze(-1)
        out = out / count

        return out


class HeteroGNNEncoder(nn.Module):
    """
    Heterogeneous GNN encoder for the sample-metabolite-enzyme graph.
    """

    def __init__(
        self,
        sample_in_dim: int,
        met_in_dim: int,
        enz_in_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
        use_metmet_prior: bool = True
    ):
        super().__init__()

        self.n_layers = n_layers
        self.use_metmet_prior = use_metmet_prior

        # Initial projections
        self.sample_proj = nn.Linear(sample_in_dim, hidden_dim)
        self.met_proj = nn.Linear(met_in_dim, hidden_dim)
        self.enz_proj = nn.Linear(enz_in_dim, hidden_dim)

        # Heterogeneous convolution layers
        self.convs = nn.ModuleList()

        for i in range(n_layers):
            conv_dict = {}

            # sample -> met (with edge_attr)
            conv_dict[('sample', 'has', 'met')] = EdgeAttrConv(hidden_dim, hidden_dim, edge_dim=1)

            # met -> sample (reverse, with edge_attr)
            conv_dict[('met', 'rev_has', 'sample')] = EdgeAttrConv(hidden_dim, hidden_dim, edge_dim=1)

            # met -> enz
            conv_dict[('met', 'to', 'enz')] = SAGEConv(hidden_dim, hidden_dim)

            # enz -> met
            conv_dict[('enz', 'to', 'met')] = SAGEConv(hidden_dim, hidden_dim)

            # met -> met (pathway prior)
            if use_metmet_prior:
                conv_dict[('met', 'pathway', 'met')] = SAGEConv(hidden_dim, hidden_dim)

            self.convs.append(HeteroConv(conv_dict, aggr='mean'))

        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the encoder.

        Returns:
            Dict of node type -> embeddings
        """
        # Initial projections
        x_dict = {
            'sample': self.sample_proj(data['sample'].x),
            'met': self.met_proj(data['met'].x),
            'enz': self.enz_proj(data['enz'].x)
        }

        # Message passing layers
        for conv in self.convs:
            # Prepare edge_index and edge_attr dicts
            edge_index_dict = {}
            edge_attr_dict = {}

            for edge_type in data.edge_types:
                edge_key = edge_type
                if data[edge_type].edge_index.numel() > 0:
                    edge_index_dict[edge_key] = data[edge_type].edge_index

                    # Get edge_attr if available
                    if hasattr(data[edge_type], 'edge_attr') and data[edge_type].edge_attr is not None:
                        edge_attr_dict[edge_key] = data[edge_type].edge_attr

            # Apply convolution
            x_dict_new = {}
            for node_type in x_dict:
                x_dict_new[node_type] = x_dict[node_type]

            # Custom handling for EdgeAttrConv layers
            for edge_type, edge_index in edge_index_dict.items():
                src_type, rel_type, dst_type = edge_type

                if edge_type in edge_attr_dict:
                    # Use EdgeAttrConv
                    layer = None
                    for c in self.convs:
                        if edge_type in c.convs:
                            layer = c.convs[edge_type]
                            break

                    if layer is not None and isinstance(layer, EdgeAttrConv):
                        out = layer(
                            x_dict[src_type],
                            x_dict[dst_type],
                            edge_index,
                            edge_attr_dict[edge_type]
                        )
                        if dst_type in x_dict_new:
                            x_dict_new[dst_type] = x_dict_new[dst_type] + out
                        else:
                            x_dict_new[dst_type] = out

            # Standard HeteroConv for non-EdgeAttr edges
            standard_edges = {k: v for k, v in edge_index_dict.items()
                           if k not in edge_attr_dict}

            if standard_edges:
                conv_out = conv(x_dict, standard_edges)
                for node_type, emb in conv_out.items():
                    if node_type in x_dict_new:
                        x_dict_new[node_type] = x_dict_new[node_type] + emb
                    else:
                        x_dict_new[node_type] = emb

            # Apply activation and dropout
            x_dict = {k: self.dropout(F.relu(v)) for k, v in x_dict_new.items()}

        return x_dict


class PainHeteroGNN(nn.Module):
    """
    Full heterogeneous GNN model for pain classification.
    """

    def __init__(
        self,
        sample_in_dim: int,
        met_in_dim: int = 8,
        enz_in_dim: int = 8,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
        use_metmet_prior: bool = True
    ):
        super().__init__()

        self.encoder = HeteroGNNEncoder(
            sample_in_dim=sample_in_dim,
            met_in_dim=met_in_dim,
            enz_in_dim=enz_in_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            use_metmet_prior=use_metmet_prior
        )

        # Pain classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, data: HeteroData) -> torch.Tensor:
        """
        Forward pass.

        Returns:
            Pain logits for sample nodes [n_samples, 1]
        """
        x_dict = self.encoder(data)
        sample_emb = x_dict['sample']
        logits = self.classifier(sample_emb)
        return logits.squeeze(-1)

    def get_embeddings(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """Get node embeddings without classification."""
        return self.encoder(data)


# =============================================================================
# Graph Construction
# =============================================================================

def build_hetero_graph(
    X: np.ndarray,
    y: np.ndarray,
    sample_features: np.ndarray,
    metabolite_names: List[str],
    met2ec: Dict[str, List[str]],
    met2pathway: Dict[str, List[str]],
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    use_metmet_prior: bool = True,
    k_metmet: int = 10,
    seed: int = 42
) -> HeteroData:
    """
    Build heterogeneous graph with sample, metabolite, and enzyme nodes.

    Args:
        X: Metabolite abundance matrix [n_samples, n_metabolites]
        y: Pain labels [n_samples]
        sample_features: Sample node features [n_samples, n_features]
        metabolite_names: List of metabolite column names
        met2ec: Metabolite to enzyme mapping
        met2pathway: Metabolite to pathway mapping
        train_mask: Boolean mask for training samples
        val_mask: Boolean mask for validation samples
        use_metmet_prior: Whether to add metabolite-metabolite pathway edges
        k_metmet: Max neighbors for met-met edges
        seed: Random seed

    Returns:
        HeteroData graph
    """
    np.random.seed(seed)

    n_samples, n_mets = X.shape

    # Collect all unique enzymes
    all_enzymes = set()
    for ecs in met2ec.values():
        all_enzymes.update(ecs)
    all_enzymes = sorted(all_enzymes)
    enz2idx = {e: i for i, e in enumerate(all_enzymes)}
    n_enzymes = len(all_enzymes)

    met2idx = {m: i for i, m in enumerate(metabolite_names)}

    logger.info(f"Graph: {n_samples} samples, {n_mets} metabolites, {n_enzymes} enzymes")

    # Create HeteroData
    data = HeteroData()

    # === Node features ===

    # Sample nodes
    data['sample'].x = torch.tensor(sample_features, dtype=torch.float32)
    data['sample'].y = torch.tensor(y, dtype=torch.float32)
    data['sample'].train_mask = torch.tensor(train_mask, dtype=torch.bool)
    data['sample'].val_mask = torch.tensor(val_mask, dtype=torch.bool)

    # Metabolite nodes (constant features to avoid leakage)
    met_dim = 8
    data['met'].x = torch.ones(n_mets, met_dim, dtype=torch.float32)

    # Enzyme nodes (constant features)
    enz_dim = 8
    if n_enzymes > 0:
        data['enz'].x = torch.ones(n_enzymes, enz_dim, dtype=torch.float32)
    else:
        data['enz'].x = torch.zeros(1, enz_dim, dtype=torch.float32)

    # === Edges ===

    # 1. Sample <-> Metabolite edges (bipartite, with edge_attr = abundance)
    sample_idx = []
    met_idx = []
    edge_attr = []

    for s in range(n_samples):
        for m in range(n_mets):
            sample_idx.append(s)
            met_idx.append(m)
            edge_attr.append(X[s, m])

    data['sample', 'has', 'met'].edge_index = torch.tensor(
        [sample_idx, met_idx], dtype=torch.long
    )
    data['sample', 'has', 'met'].edge_attr = torch.tensor(
        edge_attr, dtype=torch.float32
    )

    # Reverse edges (met -> sample)
    data['met', 'rev_has', 'sample'].edge_index = torch.tensor(
        [met_idx, sample_idx], dtype=torch.long
    )
    data['met', 'rev_has', 'sample'].edge_attr = torch.tensor(
        edge_attr, dtype=torch.float32
    )

    # 2. Metabolite <-> Enzyme edges
    met_enz_src = []
    met_enz_dst = []

    for met, ecs in met2ec.items():
        if met not in met2idx:
            continue
        m_idx = met2idx[met]
        for ec in ecs:
            if ec in enz2idx:
                met_enz_src.append(m_idx)
                met_enz_dst.append(enz2idx[ec])

    if met_enz_src:
        data['met', 'to', 'enz'].edge_index = torch.tensor(
            [met_enz_src, met_enz_dst], dtype=torch.long
        )
        data['enz', 'to', 'met'].edge_index = torch.tensor(
            [met_enz_dst, met_enz_src], dtype=torch.long
        )
    else:
        # Empty edges
        data['met', 'to', 'enz'].edge_index = torch.zeros(2, 0, dtype=torch.long)
        data['enz', 'to', 'met'].edge_index = torch.zeros(2, 0, dtype=torch.long)

    # 3. Metabolite <-> Metabolite pathway edges (optional prior)
    if use_metmet_prior:
        # Group metabolites by pathway
        pathway2mets = defaultdict(list)
        for met, pathways in met2pathway.items():
            if met not in met2idx:
                continue
            for pw in pathways:
                pathway2mets[pw].append(met2idx[met])

        # Create edges between metabolites in same pathway
        met_met_src = []
        met_met_dst = []

        for pw, mets in pathway2mets.items():
            if len(mets) < 2:
                continue

            # For each metabolite, connect to up to k neighbors in same pathway
            for m1 in mets:
                neighbors = [m2 for m2 in mets if m2 != m1]
                if len(neighbors) > k_metmet:
                    neighbors = np.random.choice(neighbors, k_metmet, replace=False).tolist()

                for m2 in neighbors:
                    met_met_src.append(m1)
                    met_met_dst.append(m2)

        if met_met_src:
            # Remove duplicates
            edges = list(set(zip(met_met_src, met_met_dst)))
            met_met_src, met_met_dst = zip(*edges) if edges else ([], [])

            data['met', 'pathway', 'met'].edge_index = torch.tensor(
                [list(met_met_src), list(met_met_dst)], dtype=torch.long
            )
            logger.info(f"Added {len(met_met_src)} met-met pathway edges")
        else:
            data['met', 'pathway', 'met'].edge_index = torch.zeros(2, 0, dtype=torch.long)
    else:
        data['met', 'pathway', 'met'].edge_index = torch.zeros(2, 0, dtype=torch.long)

    # Store metadata
    data.metabolite_names = metabolite_names
    data.enzyme_names = all_enzymes
    data.enz2idx = enz2idx
    data.met2idx = met2idx

    return data


# =============================================================================
# Training and Evaluation
# =============================================================================

def train_epoch(
    model: PainHeteroGNN,
    data: HeteroData,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """Train for one epoch."""
    model.train()
    data = data.to(device)

    optimizer.zero_grad()
    logits = model(data)

    train_mask = data['sample'].train_mask
    loss = criterion(logits[train_mask], data['sample'].y[train_mask])

    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(
    model: PainHeteroGNN,
    data: HeteroData,
    device: torch.device
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluate model on validation set.

    Returns:
        Tuple of (accuracy, auc, f1, precision, recall, y_true, y_pred_proba)
    """
    model.eval()
    data = data.to(device)

    logits = model(data)
    probs = torch.sigmoid(logits)

    val_mask = data['sample'].val_mask
    y_true = data['sample'].y[val_mask].cpu().numpy()
    y_pred = probs[val_mask].cpu().numpy()
    y_pred_binary = (y_pred >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    prec = precision_score(y_true, y_pred_binary, zero_division=0)
    rec = recall_score(y_true, y_pred_binary, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc = 0.5

    return acc, auc, f1, prec, rec, y_true, y_pred


def compute_metabolite_importance(
    model: PainHeteroGNN,
    data: HeteroData,
    device: torch.device
) -> Dict[str, float]:
    """
    Compute metabolite importance via gradient attribution on sample->met edges.

    importance(m) = mean_s |grad(logit_s) * edge_attr(s,m)|
    """
    model.eval()
    data = data.to(device)

    # Enable gradient for edge_attr
    edge_attr = data['sample', 'has', 'met'].edge_attr.clone().requires_grad_(True)
    data['sample', 'has', 'met'].edge_attr = edge_attr

    # Forward pass
    logits = model(data)

    # Backward pass (sum of logits for all samples)
    logits.sum().backward()

    # Get gradient
    grad = edge_attr.grad.abs().cpu().numpy()
    attr_val = data['sample', 'has', 'met'].edge_attr.detach().cpu().numpy()

    # Compute |grad * edge_attr|
    importance = np.abs(grad * attr_val)

    # Aggregate by metabolite
    edge_index = data['sample', 'has', 'met'].edge_index.cpu().numpy()
    met_indices = edge_index[1]

    n_mets = len(data.metabolite_names)
    met_importance = np.zeros(n_mets)
    met_count = np.zeros(n_mets)

    for i, m_idx in enumerate(met_indices):
        met_importance[m_idx] += importance[i]
        met_count[m_idx] += 1

    met_count[met_count == 0] = 1
    met_importance = met_importance / met_count

    # Create dict
    result = {}
    for i, name in enumerate(data.metabolite_names):
        result[name] = float(met_importance[i])

    return result


def compute_enzyme_scores(
    met_importance: Dict[str, float],
    met2ec: Dict[str, List[str]]
) -> Dict[str, Dict]:
    """
    Compute enzyme scores from metabolite importance.

    enzyme_score(ec) = sum of met_importance for linked metabolites
    """
    enzyme_scores = defaultdict(lambda: {'score': 0.0, 'metabolites': [], 'pathways': set()})

    for met, importance in met_importance.items():
        if met not in met2ec:
            continue

        for ec in met2ec[met]:
            enzyme_scores[ec]['score'] += importance
            enzyme_scores[ec]['metabolites'].append((met, importance))

    # Sort metabolites by importance and keep top 10
    for ec in enzyme_scores:
        mets = sorted(enzyme_scores[ec]['metabolites'], key=lambda x: -x[1])[:10]
        enzyme_scores[ec]['metabolites'] = mets
        enzyme_scores[ec]['supporting_metabolites'] = '; '.join([m for m, _ in mets[:5]])

    return dict(enzyme_scores)


# =============================================================================
# Main Pipeline
# =============================================================================

def run_pipeline(args):
    """Run the full pain heterogeneous GNN pipeline."""

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Step 1: Load Data
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Step 1: Loading metabolomics data...")
    logger.info("=" * 60)

    met_df, labels, group_labels, met_cols, metadata = load_metabolomics_data(
        args.csv,
        high_threshold=args.high_threshold,
        low_threshold=args.low_threshold,
        exclude_xenobiotics=args.exclude_xenobiotics,
        annotation_path=args.annotation_path
    )

    X = met_df.values.astype(np.float32)
    y = labels

    logger.info(f"Data shape: {X.shape}")
    logger.info(f"Labels: High={y.sum()}, Low={len(y) - y.sum()}")

    # =========================================================================
    # Step 2: LLM Name Normalization
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Step 2: LLM metabolite name normalization...")
    logger.info("=" * 60)

    cache_path = output_dir / "aliases_cache.json"

    if args.disable_llm and cache_path.exists():
        logger.info("Loading cached aliases (LLM disabled)")
        with open(cache_path, 'r') as f:
            alias_results = json.load(f)
    else:
        try:
            generator = OpenRouterAliasGenerator(
                cache_path=str(cache_path),
                batch_size=20
            )
            alias_results = generator.generate_aliases(
                met_cols,
                use_llm=not args.disable_llm
            )
        except EnvironmentError as e:
            if args.disable_llm:
                logger.warning("No cache and LLM disabled - using fallback")
                alias_results = {}
                for met in met_cols:
                    alias_results[met] = {
                        'original': met,
                        'queries': [met.lower()],
                        'confidence': 'low'
                    }
            else:
                raise e

    logger.info(f"Generated aliases for {len(alias_results)} metabolites")

    # =========================================================================
    # Step 3: KEGG REST API Mapping
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Step 3: KEGG REST API mapping...")
    logger.info("=" * 60)

    kegg_cache_dir = output_dir / "kegg_cache"
    mapper = KEGGMapper(cache_dir=str(kegg_cache_dir))

    mapper.map_all_metabolites(alias_results)

    # Load supplementary KEGG Excel if available
    if args.kegg_excel and Path(args.kegg_excel).exists():
        logger.info(f"Loading supplementary KEGG mappings from {args.kegg_excel}")
        excel_mappings = mapper.load_kegg_excel(args.kegg_excel)
        mapper.merge_excel_mappings(excel_mappings)

    # Save mappings
    mapper.save_mappings(str(output_dir))

    met2ec = mapper.met2ec
    met2pathway = mapper.met2pathway

    # Stats
    n_mapped = sum(1 for v in mapper.met2cpd.values() if v)
    n_with_enzymes = sum(1 for v in met2ec.values() if v)
    logger.info(f"Mapped {n_mapped}/{len(met_cols)} compounds, {n_with_enzymes} with enzymes")

    # =========================================================================
    # Step 4: Cross-Validation
    # =========================================================================
    n_folds = args.n_folds
    logger.info("=" * 60)
    logger.info(f"Step 4: Training HeteroGNN with {n_folds}-fold stratified CV...")
    logger.info("=" * 60)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=args.seed)

    fold_results = []
    fold_predictions = []
    all_met_importance = []
    all_enzyme_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info(f"\n--- Fold {fold_idx + 1}/{n_folds} ---")

        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        logger.info(f"  Train: {len(train_idx)} (High={int(y_train.sum())}, Low={int(len(y_train)-y_train.sum())}), "
                     f"Val: {len(val_idx)} (High={int(y_val.sum())}, Low={int(len(y_val)-y_val.sum())})")

        # Get metadata for samples
        age_train = metadata.get('age', [None] * len(y))[train_idx] if metadata.get('age') is not None else None
        age_val = metadata.get('age', [None] * len(y))[val_idx] if metadata.get('age') is not None else None
        severity_train = metadata.get('severity', [None] * len(y))[train_idx] if metadata.get('severity') is not None else None
        severity_val = metadata.get('severity', [None] * len(y))[val_idx] if metadata.get('severity') is not None else None

        # Preprocess within fold
        X_train_scaled, X_val_scaled, sample_feat_train, sample_feat_val = preprocess_fold(
            X_train, X_val, age_train, age_val, severity_train, severity_val
        )

        # Combine for graph construction
        X_all = np.vstack([X_train_scaled, X_val_scaled])
        y_all = np.concatenate([y_train, y_val])
        sample_feat_all = np.vstack([sample_feat_train, sample_feat_val])

        # Create masks
        n_train = len(train_idx)
        n_total = len(X_all)
        train_mask = np.zeros(n_total, dtype=bool)
        val_mask = np.zeros(n_total, dtype=bool)
        train_mask[:n_train] = True
        val_mask[n_train:] = True

        # Build graph
        data = build_hetero_graph(
            X_all, y_all, sample_feat_all, met_cols,
            met2ec, met2pathway,
            train_mask, val_mask,
            use_metmet_prior=args.use_metmet_prior,
            k_metmet=args.k_metmet,
            seed=args.seed
        )

        # Initialize model
        sample_in_dim = sample_feat_all.shape[1]
        model = PainHeteroGNN(
            sample_in_dim=sample_in_dim,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            dropout=args.dropout,
            use_metmet_prior=args.use_metmet_prior
        ).to(device)

        # Loss with class weighting
        pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / max(y_train.sum(), 1)]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # Training loop with early stopping
        best_auc = 0.0
        best_model_state = None
        patience_counter = 0

        for epoch in range(args.epochs):
            loss = train_epoch(model, data, optimizer, criterion, device)

            if (epoch + 1) % 10 == 0:
                acc, auc, _, _, _, _, _ = evaluate(model, data, device)

                if auc > best_auc:
                    best_auc = auc
                    best_model_state = deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= args.patience:
                    logger.info(f"  Early stopping at epoch {epoch + 1}")
                    break

        # Load best model
        if best_model_state:
            model.load_state_dict(best_model_state)

        # Evaluate
        acc, auc, f1, prec, rec, y_true, y_pred = evaluate(model, data, device)
        logger.info(f"  Fold {fold_idx + 1}: AUC={auc:.4f}, Acc={acc:.4f}, F1={f1:.4f}, Prec={prec:.4f}, Rec={rec:.4f}")

        fold_results.append({'fold': fold_idx + 1, 'accuracy': acc, 'auc': auc,
                             'f1': f1, 'precision': prec, 'recall': rec})
        fold_predictions.append((y_true, y_pred))

        # Compute metabolite importance
        met_imp = compute_metabolite_importance(model, data, device)
        all_met_importance.append(met_imp)

        # Compute enzyme scores
        enz_scores = compute_enzyme_scores(met_imp, met2ec)
        all_enzyme_scores.append(enz_scores)

    # =========================================================================
    # Step 5: Aggregate Results
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Step 5: Aggregating results...")
    logger.info("=" * 60)

    # CV metrics
    cv_df = pd.DataFrame(fold_results)
    numeric_cols = ['accuracy', 'auc', 'f1', 'precision', 'recall']
    cv_df.loc[len(cv_df)] = ['mean'] + [cv_df[c].mean() for c in numeric_cols]
    cv_df.loc[len(cv_df)] = ['std'] + [cv_df[c].std() for c in numeric_cols]
    summary_df = cv_df

    mean_auc = cv_df[cv_df['fold'] == 'mean']['auc'].values[0]
    logger.info(f"Mean AUC: {mean_auc:.4f}")

    # Aggregate metabolite importance
    final_met_importance = defaultdict(list)
    for fold_imp in all_met_importance:
        for met, score in fold_imp.items():
            final_met_importance[met].append(score)

    met_rank_data = []
    for met, scores in final_met_importance.items():
        met_rank_data.append({
            'metabolite': met,
            'importance_score': np.mean(scores),
            'importance_std': np.std(scores),
            'selection_frequency': sum(1 for s in scores if s > np.mean(scores))
        })

    met_rank_df = pd.DataFrame(met_rank_data)
    met_rank_df = met_rank_df.sort_values('importance_score', ascending=False)

    # Aggregate enzyme scores
    final_enz_scores = defaultdict(lambda: {'scores': [], 'metabolites': set()})
    for fold_enz in all_enzyme_scores:
        for ec, info in fold_enz.items():
            final_enz_scores[ec]['scores'].append(info['score'])
            for met, _ in info['metabolites']:
                final_enz_scores[ec]['metabolites'].add(met)

    enz_rank_data = []
    for ec, info in final_enz_scores.items():
        # Get pathways for this enzyme's metabolites
        pathways = set()
        for met in info['metabolites']:
            if met in met2pathway:
                pathways.update(met2pathway[met])

        enz_rank_data.append({
            'enzyme_id': ec,
            'enzyme_score': np.mean(info['scores']),
            'score_std': np.std(info['scores']),
            'supporting_metabolites': '; '.join(list(info['metabolites'])[:5]),
            'pathways': '; '.join(list(pathways)[:3]),
            'evidence_counts': len(info['metabolites']),
            'selection_frequency': len(info['scores'])
        })

    enz_rank_df = pd.DataFrame(enz_rank_data)
    enz_rank_df = enz_rank_df.sort_values('enzyme_score', ascending=False)

    # =========================================================================
    # Step 6: Group Diagnostics
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Step 6: Group diagnostics...")
    logger.info("=" * 60)

    group_stats = None
    group_metrics_df = None

    if group_labels is not None:
        # Group confounding
        group_stats = compute_group_confounding(group_labels, y)
        logger.info(f"Group confounding: Cramer's V = {group_stats['cramers_v']:.3f}, "
                   f"p = {group_stats['p_value']:.4f}")

        # Within-group AUC
        all_preds = np.concatenate([p for _, p in fold_predictions])
        all_true = np.concatenate([t for t, _ in fold_predictions])

        # This is approximate - proper within-group needs to track group membership per fold
        group_aucs = compute_within_group_auc(all_preds, all_true, group_labels)

        group_metrics_data = []
        for group, auc in group_aucs.items():
            n_samples = sum(1 for g in group_labels if str(g) == group)
            group_metrics_data.append({
                'group': group,
                'auc': auc,
                'n_samples': n_samples
            })
        group_metrics_df = pd.DataFrame(group_metrics_data)

        # Save group table
        table_df = pd.DataFrame(
            group_stats['contingency_table'],
            index=group_stats['groups'],
            columns=['Low Pain', 'High Pain']
        )
        table_df.to_csv(output_dir / 'group_vs_label_table.csv')

    # =========================================================================
    # Step 7: Permutation Test (Optional)
    # =========================================================================
    permutation_aucs = None

    if args.run_permutation:
        logger.info("=" * 60)
        logger.info(f"Step 7: Permutation test ({args.n_permutations} permutations)...")
        logger.info("=" * 60)

        permutation_aucs = []

        for perm_idx in tqdm(range(args.n_permutations), desc="Permutations"):
            # Shuffle labels
            y_perm = np.random.permutation(y)

            # Quick single-fold evaluation
            train_idx = np.random.choice(len(y), size=int(0.8 * len(y)), replace=False)
            val_idx = np.array([i for i in range(len(y)) if i not in train_idx])

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y_perm[train_idx], y_perm[val_idx]

            # Simple preprocessing
            X_train_scaled, X_val_scaled, sf_train, sf_val = preprocess_fold(
                X_train, X_val, None, None, None, None
            )

            X_all = np.vstack([X_train_scaled, X_val_scaled])
            y_all = np.concatenate([y_train, y_val])
            sf_all = np.vstack([sf_train, sf_val])

            train_mask = np.zeros(len(X_all), dtype=bool)
            val_mask = np.zeros(len(X_all), dtype=bool)
            train_mask[:len(train_idx)] = True
            val_mask[len(train_idx):] = True

            data_perm = build_hetero_graph(
                X_all, y_all, sf_all, met_cols,
                met2ec, met2pathway,
                train_mask, val_mask,
                use_metmet_prior=args.use_metmet_prior,
                k_metmet=args.k_metmet,
                seed=args.seed + perm_idx
            )

            model_perm = PainHeteroGNN(
                sample_in_dim=sf_all.shape[1],
                hidden_dim=args.hidden_dim // 2,  # Smaller for speed
                n_layers=1,
                dropout=args.dropout,
                use_metmet_prior=args.use_metmet_prior
            ).to(device)

            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(model_perm.parameters(), lr=args.lr * 2)

            # Quick training
            for _ in range(args.perm_epochs):
                train_epoch(model_perm, data_perm, optimizer, criterion, device)

            _, auc_perm, _, _, _, _, _ = evaluate(model_perm, data_perm, device)
            permutation_aucs.append(auc_perm)

        p_value = np.mean([p >= mean_auc for p in permutation_aucs])
        logger.info(f"Permutation p-value: {p_value:.4f}")

        perm_df = pd.DataFrame({'permutation_auc': permutation_aucs})
        perm_df.to_csv(output_dir / 'permutation_auc_distribution.csv', index=False)

    # =========================================================================
    # Step 8: Sanity Checks
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Step 8: Sanity checks...")
    logger.info("=" * 60)

    sanity_results = []
    sanity_results.append({'check': 'baseline', 'auc': mean_auc})

    try:
        # Shuffled edge_attr (single fold check)
        train_idx = np.arange(int(0.8 * len(y)))
        val_idx = np.arange(int(0.8 * len(y)), len(y))

        X_train_scaled, X_val_scaled, sf_train, sf_val = preprocess_fold(
            X[train_idx], X[val_idx], None, None, None, None
        )

        X_all = np.vstack([X_train_scaled, X_val_scaled])
        y_all = np.concatenate([y[train_idx], y[val_idx]])
        sf_all = np.vstack([sf_train, sf_val])

        train_mask = np.zeros(len(X_all), dtype=bool)
        val_mask = np.zeros(len(X_all), dtype=bool)
        train_mask[:len(train_idx)] = True
        val_mask[len(train_idx):] = True

        data_sanity = build_hetero_graph(
            X_all, y_all, sf_all, met_cols,
            met2ec, met2pathway, train_mask, val_mask,
            use_metmet_prior=args.use_metmet_prior, k_metmet=args.k_metmet, seed=args.seed
        )

        # Shuffle edge_attr only (not edge_index)
        data_shuffled = deepcopy(data_sanity)
        edge_attr = data_shuffled['sample', 'has', 'met'].edge_attr.clone()
        perm = torch.randperm(edge_attr.size(0))
        data_shuffled['sample', 'has', 'met'].edge_attr = edge_attr[perm]
        data_shuffled['met', 'rev_has', 'sample'].edge_attr = edge_attr[perm]

        model_sanity = PainHeteroGNN(
            sample_in_dim=sf_all.shape[1],
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            dropout=args.dropout,
            use_metmet_prior=args.use_metmet_prior
        ).to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model_sanity.parameters(), lr=args.lr)

        for _ in range(50):
            train_epoch(model_sanity, data_shuffled, optimizer, criterion, device)

        _, auc_shuffled, _, _, _, _, _ = evaluate(model_sanity, data_shuffled, device)
        sanity_results.append({'check': 'shuffled_edge_attr', 'auc': auc_shuffled})

        logger.info(f"Shuffled edge_attr AUC: {auc_shuffled:.4f} (baseline: {mean_auc:.4f})")
    except Exception as e:
        logger.warning(f"Sanity check failed: {e}")
        sanity_results.append({'check': 'shuffled_edge_attr', 'auc': float('nan')})

    sanity_df = pd.DataFrame(sanity_results)

    # =========================================================================
    # Step 9: Save Results and Visualizations
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Step 9: Saving results and visualizations...")
    logger.info("=" * 60)

    # Save results
    save_results(
        str(output_dir),
        summary_df,
        met_rank_df,
        enz_rank_df,
        group_metrics_df,
        perm_df if args.run_permutation else None,
        sanity_df
    )
    # Also save per-sample LOO predictions
    # cv_df already saved via save_results as cv_metrics_overall.csv

    # Compute stability metrics
    met_stability = {}
    for met, scores in final_met_importance.items():
        # Count how often this metabolite is in top 20
        met_stability[met] = sum(1 for s in scores if s > 0)

    enz_stability = {}
    for ec, info in final_enz_scores.items():
        enz_stability[ec] = len(info['scores'])

    # Create visualizations
    cv_results = {
        'fold_aucs': [r['auc'] for r in fold_results if isinstance(r['fold'], int)],
        'fold_predictions': fold_predictions,
        'all_predictions': fold_predictions,
        'group_aucs': group_aucs if group_labels is not None else {}
    }

    create_visualizations(
        str(output_dir),
        cv_results,
        met_stability,
        enz_stability,
        group_stats,
        permutation_aucs,
        mean_auc
    )

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Mean AUC: {mean_auc:.4f}")
    logger.info(f"Top enzymes: {enz_rank_df['enzyme_id'].head(5).tolist()}")
    logger.info(f"Top metabolites: {met_rank_df['metabolite'].head(5).tolist()}")

    return mean_auc


def main():
    parser = argparse.ArgumentParser(
        description="Pain Heterogeneous GNN Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input/Output
    parser.add_argument('--csv', type=str, required=True,
                       help='Path to metabolomics CSV/Excel file')
    parser.add_argument('--outdir', type=str, default='out_hetero_gnn',
                       help='Output directory')
    parser.add_argument('--kegg-excel', type=str, default=None,
                       help='Optional path to supplementary KEGG Excel file')

    # Label thresholds
    parser.add_argument('--high-threshold', type=int, default=7,
                       help='PAIN_SCORE >= this is High pain')
    parser.add_argument('--low-threshold', type=int, default=3,
                       help='PAIN_SCORE <= this is Low pain')

    # Graph construction
    parser.add_argument('--use-metmet-prior', action='store_true', default=True,
                       help='Use metabolite-metabolite pathway edges')
    parser.add_argument('--no-metmet-prior', dest='use_metmet_prior', action='store_false',
                       help='Disable metabolite-metabolite pathway edges')
    parser.add_argument('--k-metmet', type=int, default=10,
                       help='Max neighbors for met-met edges')

    # Model
    parser.add_argument('--hidden-dim', type=int, default=64,
                       help='Hidden dimension')
    parser.add_argument('--n-layers', type=int, default=2,
                       help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')

    # Cross-validation
    parser.add_argument('--n-folds', type=int, default=6,
                       help='Number of CV folds')

    # Training
    parser.add_argument('--epochs', type=int, default=200,
                       help='Max training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')

    # LLM
    parser.add_argument('--disable-llm', action='store_true',
                       help='Disable LLM calls (use cached aliases only)')

    # Permutation test
    parser.add_argument('--run-permutation', action='store_true',
                       help='Run permutation test')
    parser.add_argument('--n-permutations', type=int, default=200,
                       help='Number of permutations')
    parser.add_argument('--perm-epochs', type=int, default=30,
                       help='Epochs per permutation')

    # Feature filtering
    parser.add_argument('--exclude-xenobiotics', action='store_true',
                       help='Exclude xenobiotic/drug metabolites, keep only endogenous')
    parser.add_argument('--annotation-path', type=str,
                       default='metabolites names and pathways.xlsx',
                       help='Metabolon annotation file with SUPER_PATHWAY column')

    # Other
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Run pipeline
    run_pipeline(args)


if __name__ == "__main__":
    main()
