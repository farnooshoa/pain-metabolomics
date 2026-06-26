#!/usr/bin/env python3
"""
L1-sparse Pain HeteroGNN: adds a per-metabolite learnable gate with L1
penalty, forcing the model to concentrate predictive signal on few
metabolites. Runs 6-fold stratified CV + per-feature permutation test
on the sparse model.

Usage:
    python l1_sparse_pipeline.py \
        --csv starting_template.csv \
        --outdir pain_hetero_pipeline/out_hetero_endogenous_6fold_L1 \
        --input-dir pain_hetero_pipeline/out_hetero_endogenous_6fold \
        --l1-lambda 1e-3 --n-perms 200 --seed 42
"""

import os
import sys
import json
import argparse
import logging
import random
import time
from pathlib import Path
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from pain_hetero_pipeline import (
    PainHeteroGNN, HeteroGNNEncoder,
    build_hetero_graph, train_epoch, evaluate,
    compute_metabolite_importance,
    compute_enzyme_scores,
)
from utils import load_metabolomics_data, preprocess_fold

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('l1_sparse')


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class L1SparseHeteroGNN(nn.Module):
    """Wraps PainHeteroGNN with a learnable per-metabolite ReLU-gate.

    Gate = ReLU(raw_gate). L1 penalty on gate pushes unused metabolites
    to exactly zero, producing true sparsity.
    """

    def __init__(self, n_mets, sample_in_dim, hidden_dim=64, n_layers=2,
                 dropout=0.3, use_metmet_prior=False):
        super().__init__()
        self.n_mets = n_mets
        # Init at moderate value; L1 pushes down, signal preserves some
        self.raw_gate = nn.Parameter(torch.full((n_mets,), 0.5))
        self.base = PainHeteroGNN(
            sample_in_dim=sample_in_dim, hidden_dim=hidden_dim,
            n_layers=n_layers, dropout=dropout,
            use_metmet_prior=use_metmet_prior,
        )

    @property
    def gates(self):
        return F.relu(self.raw_gate)

    def forward(self, data):
        # Apply gate to met node features *in place for this forward*
        gates = self.gates.to(data['met'].x.device)
        orig = data['met'].x
        data['met'].x = orig * gates.unsqueeze(-1)
        try:
            logits = self.base(data)
        finally:
            data['met'].x = orig
        return logits

    def get_embeddings(self, data):
        gates = self.gates.to(data['met'].x.device)
        orig = data['met'].x
        data['met'].x = orig * gates.unsqueeze(-1)
        try:
            emb = self.base.get_embeddings(data)
        finally:
            data['met'].x = orig
        return emb

    def l1_penalty(self):
        return self.gates.abs().sum()


def train_epoch_l1(model, data, optimizer, criterion, device, l1_lambda):
    model.train()
    optimizer.zero_grad()
    data = data.to(device)
    logits = model(data)
    train_mask = data['sample'].train_mask
    labels = data['sample'].y.float()
    ce = criterion(logits[train_mask], labels[train_mask])
    l1 = model.l1_penalty()
    loss = ce + l1_lambda * l1
    loss.backward()
    optimizer.step()
    return float(loss.item()), float(ce.item()), float(l1.item())


def evaluate_l1(model, data, device):
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        logits = model(data)
        probs = torch.sigmoid(logits).cpu().numpy()
    val_mask = data['sample'].val_mask.cpu().numpy()
    labels = data['sample'].y.cpu().numpy()
    y_true = labels[val_mask]
    y_pred = probs[val_mask]
    try:
        auc = roc_auc_score(y_true, y_pred)
    except Exception:
        auc = 0.5
    y_bin = (y_pred > 0.5).astype(int)
    acc = accuracy_score(y_true, y_bin)
    f1 = f1_score(y_true, y_bin, zero_division=0)
    prec = precision_score(y_true, y_bin, zero_division=0)
    rec = recall_score(y_true, y_bin, zero_division=0)
    return acc, auc, f1, prec, rec, y_true, y_pred


def gate_importance(model):
    """Importance = gate value (sigmoid of learnable parameter)."""
    return model.gates.detach().cpu().numpy()


def run_one_cv(X, y, metadata, met_cols, met2ec, met2pathway, args, device):
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True,
                          random_state=args.seed)
    met_accum = defaultdict(list)
    enz_accum = defaultdict(list)
    aucs = []
    active_counts = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        age = metadata.get('age')
        sev = metadata.get('severity')
        age_t = age[train_idx] if age is not None else None
        age_v = age[val_idx] if age is not None else None
        sev_t = sev[train_idx] if sev is not None else None
        sev_v = sev[val_idx] if sev is not None else None

        X_ts, X_vs, sf_t, sf_v = preprocess_fold(
            X_train, X_val, age_t, age_v, sev_t, sev_v
        )
        X_all = np.vstack([X_ts, X_vs])
        y_all = np.concatenate([y_train, y_val])
        sf_all = np.vstack([sf_t, sf_v])

        n_tr = len(train_idx)
        n_total = len(X_all)
        tr_mask = np.zeros(n_total, dtype=bool)
        vl_mask = np.zeros(n_total, dtype=bool)
        tr_mask[:n_tr] = True
        vl_mask[n_tr:] = True

        data = build_hetero_graph(
            X_all, y_all, sf_all, met_cols, met2ec, met2pathway,
            tr_mask, vl_mask,
            use_metmet_prior=args.use_metmet_prior,
            k_metmet=args.k_metmet, seed=args.seed,
        )

        model = L1SparseHeteroGNN(
            n_mets=len(met_cols),
            sample_in_dim=sf_all.shape[1],
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            dropout=args.dropout,
            use_metmet_prior=args.use_metmet_prior,
        ).to(device)

        pos_w = torch.tensor(
            [(len(y_train) - y_train.sum()) / max(y_train.sum(), 1)]
        ).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)

        # Full training; L1 needs many epochs to converge (no AUC-based early stop)
        for epoch in range(args.epochs):
            train_epoch_l1(model, data, opt, criterion, device, args.l1_lambda)

        _, auc, *_ = evaluate_l1(model, data, device)
        aucs.append(auc)

        gates = gate_importance(model)
        active = int((gates > 0.01).sum())
        active_counts.append(active)
        if fold_idx == 0:
            logger.info(
                f'  Fold {fold_idx+1}: AUC={auc:.3f}, active={active}/{len(met_cols)}, '
                f'gate range=[{gates.min():.4f}, {gates.max():.4f}]'
            )

        # Metabolite importance = gate value (direct, interpretable)
        for i, m in enumerate(met_cols):
            met_accum[m].append(float(gates[i]))

        # Enzyme score = sum of gate values of supporting metabolites
        enz_scores_local = defaultdict(float)
        for i, m in enumerate(met_cols):
            gval = float(gates[i])
            if gval <= 0:
                continue
            for ec in met2ec.get(m, []):
                enz_scores_local[ec] += gval
        for ec, s in enz_scores_local.items():
            enz_accum[ec].append(s)

    met_mean = {m: float(np.mean(v)) for m, v in met_accum.items()}
    enz_mean = {e: float(np.mean(v)) for e, v in enz_accum.items()}
    # Normalize
    mt = sum(met_mean.values())
    if mt > 0: met_mean = {k: v / mt for k, v in met_mean.items()}
    et = sum(enz_mean.values())
    if et > 0: enz_mean = {k: v / et for k, v in enz_mean.items()}
    return (met_mean, enz_mean,
            float(np.mean(aucs)), float(np.mean(active_counts)))


def bh_fdr(pvals):
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(n)
    out[order] = q
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--outdir', required=True)
    p.add_argument('--input-dir', required=True,
                   help='Directory containing met2ec.json, met2pathway.json')
    p.add_argument('--l1-lambda', type=float, default=1e-3)
    p.add_argument('--n-perms', type=int, default=200)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n-folds', type=int, default=6)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--hidden-dim', type=int, default=64)
    p.add_argument('--n-layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--lr', type=float, default=0.005)
    p.add_argument('--weight-decay', type=float, default=5e-4)
    p.add_argument('--high-threshold', type=int, default=4)
    p.add_argument('--low-threshold', type=int, default=3)
    p.add_argument('--exclude-xenobiotics', action='store_true')
    p.add_argument('--annotation-path', type=str,
                   default='metabolites names and pathways.xlsx')
    p.add_argument('--use-metmet-prior', action='store_true')
    p.add_argument('--k-metmet', type=int, default=5)
    p.add_argument('--fdr-alpha', type=float, default=0.05)
    p.add_argument('--skip-perms', action='store_true',
                   help='Only run real training, skip permutations')
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    input_dir = Path(args.input_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}, L1 lambda: {args.l1_lambda}')

    # Load data
    met_df, labels, group_labels, met_cols, metadata = load_metabolomics_data(
        args.csv, high_threshold=args.high_threshold,
        low_threshold=args.low_threshold,
        exclude_xenobiotics=args.exclude_xenobiotics,
        annotation_path=args.annotation_path,
    )
    X = met_df.values.astype(np.float32)
    y = np.asarray(labels)
    logger.info(f'Loaded: X={X.shape}, mets={len(met_cols)}')

    with open(input_dir / 'met2ec.json') as f:
        met2ec = json.load(f)
    with open(input_dir / 'met2pathway.json') as f:
        met2pathway = json.load(f)

    # -------- REAL RUN --------
    set_all_seeds(args.seed)
    logger.info('=' * 60)
    logger.info('Real (unshuffled) run:')
    logger.info('=' * 60)
    real_met, real_enz, real_auc, real_active = run_one_cv(
        X, y, metadata, met_cols, met2ec, met2pathway, args, device
    )
    logger.info(
        f'Real AUC={real_auc:.3f}, avg active gates={real_active:.0f}'
    )

    # Save real rankings
    real_met_df = pd.DataFrame([
        {'metabolite': m, 'sparse_score': s} for m, s in real_met.items()
    ]).sort_values('sparse_score', ascending=False)
    real_met_df.to_csv(outdir / 'metabolite_rank_sparse.csv', index=False)

    real_enz_df = pd.DataFrame([
        {'enzyme_id': e, 'sparse_score': s} for e, s in real_enz.items()
    ]).sort_values('sparse_score', ascending=False)
    real_enz_df.to_csv(outdir / 'enzyme_rank_sparse.csv', index=False)

    logger.info('Top 15 metabolites (sparse):')
    for _, r in real_met_df.head(15).iterrows():
        logger.info(f"  {r['metabolite']:50s} {r['sparse_score']:.4e}")
    logger.info('Top 15 enzymes (sparse):')
    for _, r in real_enz_df.head(15).iterrows():
        logger.info(f"  {r['enzyme_id']:15s} {r['sparse_score']:.4e}")

    if args.skip_perms:
        logger.info('Skipping permutations (--skip-perms)')
        return

    # -------- PERMUTATIONS --------
    logger.info('=' * 60)
    logger.info(f'Permutation test ({args.n_perms} perms):')
    logger.info('=' * 60)
    met_null = defaultdict(list)
    enz_null = defaultdict(list)
    perm_aucs = []

    t0 = time.time()
    for i in range(args.n_perms):
        y_perm = np.random.permutation(y)
        try:
            m_mean, e_mean, auc, _ = run_one_cv(
                X, y_perm, metadata, met_cols, met2ec, met2pathway,
                args, device
            )
        except Exception as exc:
            logger.warning(f'perm {i} failed: {exc}')
            continue
        perm_aucs.append(auc)
        for m, s in m_mean.items():
            met_null[m].append(s)
        for e, s in e_mean.items():
            enz_null[e].append(s)

        if (i + 1) % 5 == 0 or i == 0:
            el = time.time() - t0
            eta = el / (i + 1) * (args.n_perms - i - 1)
            logger.info(
                f'perm {i + 1}/{args.n_perms} | auc={auc:.3f} | '
                f'elapsed={el/60:.1f}m | ETA={eta/60:.1f}m'
            )

    logger.info(
        f'Perm AUC: mean={np.mean(perm_aucs):.3f}, std={np.std(perm_aucs):.3f}'
    )

    # -------- P-VALUES --------
    def compute_p(real_map, null_map, key_name):
        rows = []
        for f in sorted(real_map.keys()):
            real = real_map[f]
            null = np.array(null_map.get(f, []))
            if len(null) == 0:
                p = np.nan
            else:
                p = (1 + int((null >= real).sum())) / (len(null) + 1)
            rows.append({key_name: f, 'real_score': real,
                         'n_null': len(null), 'p_value': p})
        df = pd.DataFrame(rows)
        valid = df['p_value'].notna()
        q = np.full(len(df), np.nan)
        if valid.any():
            q[valid.values] = bh_fdr(df.loc[valid, 'p_value'].values)
        df['q_value'] = q
        df['significant_fdr'] = df['q_value'] < args.fdr_alpha
        return df.sort_values('p_value')

    met_p = compute_p(real_met, met_null, 'metabolite')
    met_p.to_csv(outdir / 'permutation_pvalues_met_sparse.csv', index=False)
    enz_p = compute_p(real_enz, enz_null, 'enzyme_id')
    enz_p.to_csv(outdir / 'permutation_pvalues_enz_sparse.csv', index=False)

    n_sig_m = int(met_p['significant_fdr'].sum())
    n_sig_e = int(enz_p['significant_fdr'].sum())
    logger.info(
        f'Significant (FDR<{args.fdr_alpha}): '
        f'mets={n_sig_m}/{len(met_p)}, enzymes={n_sig_e}/{len(enz_p)}'
    )
    logger.info('Top 10 mets by p-value:')
    for _, r in met_p.head(10).iterrows():
        logger.info(
            f"  {r['metabolite']:50s} p={r['p_value']:.4f} "
            f"q={r['q_value']:.4f}"
        )
    logger.info('Top 10 enzymes by p-value:')
    for _, r in enz_p.head(10).iterrows():
        logger.info(
            f"  {r['enzyme_id']:15s} p={r['p_value']:.4f} "
            f"q={r['q_value']:.4f}"
        )


if __name__ == '__main__':
    main()
