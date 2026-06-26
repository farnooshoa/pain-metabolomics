#!/usr/bin/env python3
"""
Pathway-level permutation test.

Aggregates metabolite importance scores into Metabolon sub-pathways
(mean of metabolites in that pathway, normalized to proportions per run),
then computes empirical p-values and BH-FDR q-values against the
label-shuffled null distribution.

Pathway-level aggregation increases statistical power by pooling evidence
from multiple metabolites within the same biological grouping.

Usage:
    python permutation_test_pathway.py \
        --csv starting_template.csv \
        --outdir pain_hetero_pipeline/out_hetero_endogenous_6fold \
        --n-perms 200 --seed 42
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
from sklearn.model_selection import StratifiedKFold

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from pain_hetero_pipeline import (
    PainHeteroGNN, build_hetero_graph, train_epoch, evaluate,
    compute_metabolite_importance,
)
from utils import load_metabolomics_data, preprocess_fold

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('perm_pathway')


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def aggregate_to_pathways(met_scores: dict, met2sp: dict) -> dict:
    """Aggregate metabolite scores to sub-pathway scores via MEAN."""
    sp_scores = defaultdict(list)
    for met, score in met_scores.items():
        sp = met2sp.get(met)
        if sp is not None and not (isinstance(sp, float) and np.isnan(sp)):
            sp_scores[sp].append(score)
    return {sp: float(np.mean(v)) for sp, v in sp_scores.items()}


def normalize_to_proportions(scores: dict) -> dict:
    total = sum(scores.values())
    if total <= 0:
        return scores
    return {k: v / total for k, v in scores.items()}


def run_one_cv(X, y, metadata, met_cols, met2ec, met2pathway, met2sp,
               args, device):
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True,
                          random_state=args.seed)
    met_accum = defaultdict(list)
    aucs = []

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

        model = PainHeteroGNN(
            sample_in_dim=sf_all.shape[1], hidden_dim=args.hidden_dim,
            n_layers=args.n_layers, dropout=args.dropout,
            use_metmet_prior=args.use_metmet_prior,
        ).to(device)

        pos_w = torch.tensor(
            [(len(y_train) - y_train.sum()) / max(y_train.sum(), 1)]
        ).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)

        best_auc = 0.0
        best_state = None
        patience = 0
        for epoch in range(args.epochs):
            train_epoch(model, data, opt, criterion, device)
            if (epoch + 1) % 10 == 0:
                _, auc, *_ = evaluate(model, data, device)
                if auc > best_auc:
                    best_auc = auc
                    best_state = deepcopy(model.state_dict())
                    patience = 0
                else:
                    patience += 1
                    if patience >= args.patience:
                        break
        if best_state is not None:
            model.load_state_dict(best_state)

        _, auc, *_ = evaluate(model, data, device)
        aucs.append(auc)
        for m, s in compute_metabolite_importance(model, data, device).items():
            met_accum[m].append(s)

    met_mean = {m: float(np.mean(v)) for m, v in met_accum.items()}
    sp_mean = aggregate_to_pathways(met_mean, met2sp)
    sp_norm = normalize_to_proportions(sp_mean)
    return sp_norm, float(np.mean(aucs))


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
    p.add_argument('--min-pathway-size', type=int, default=3,
                   help='Ignore sub-pathways with fewer members')
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    # Load data
    met_df, labels, group_labels, met_cols, metadata = load_metabolomics_data(
        args.csv, high_threshold=args.high_threshold,
        low_threshold=args.low_threshold,
        exclude_xenobiotics=args.exclude_xenobiotics,
        annotation_path=args.annotation_path,
    )
    X = met_df.values.astype(np.float32)
    y = np.asarray(labels)
    logger.info(f'Loaded: X={X.shape}, y={y.shape}, mets={len(met_cols)}')

    # Load metabolite -> sub-pathway map
    annot = pd.read_excel(args.annotation_path, sheet_name='Chemical Annotation')
    met2sp = dict(zip(annot['CHEMICAL_NAME'], annot['SUB_PATHWAY']))

    # Filter small pathways
    sp_sizes = defaultdict(int)
    for m in met_cols:
        sp = met2sp.get(m)
        if sp and not (isinstance(sp, float) and np.isnan(sp)):
            sp_sizes[sp] += 1
    kept_sps = {sp for sp, n in sp_sizes.items() if n >= args.min_pathway_size}
    logger.info(
        f'Sub-pathways: {len(sp_sizes)} total, '
        f'{len(kept_sps)} with >= {args.min_pathway_size} members'
    )

    # Load KEGG maps
    with open(outdir / 'met2ec.json') as f:
        met2ec = json.load(f)
    with open(outdir / 'met2pathway.json') as f:
        met2pathway = json.load(f)

    # Load real metabolite scores and aggregate
    real_met_df = pd.read_csv(outdir / 'metabolite_rank.csv')
    real_met = dict(zip(real_met_df['metabolite'],
                        real_met_df['importance_score']))
    real_sp = aggregate_to_pathways(real_met, met2sp)
    real_sp = normalize_to_proportions(real_sp)
    logger.info(f'Real sub-pathway scores: {len(real_sp)}')

    # Run permutations
    set_all_seeds(args.seed)
    sp_null = defaultdict(list)
    perm_aucs = []

    t0 = time.time()
    for i in range(args.n_perms):
        y_perm = np.random.permutation(y)
        try:
            sp_scores, auc = run_one_cv(
                X, y_perm, metadata, met_cols, met2ec, met2pathway, met2sp,
                args, device
            )
        except Exception as exc:
            logger.warning(f'perm {i} failed: {exc}')
            continue
        perm_aucs.append(auc)
        for sp, s in sp_scores.items():
            sp_null[sp].append(s)

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

    # Compute p-values only for kept sub-pathways
    rows = []
    for sp in sorted(real_sp.keys()):
        if sp not in kept_sps:
            continue
        real = real_sp[sp]
        null = np.array(sp_null.get(sp, []))
        if len(null) == 0:
            p = np.nan
        else:
            p = (1 + int((null >= real).sum())) / (len(null) + 1)
        rows.append({
            'sub_pathway': sp,
            'n_metabolites': sp_sizes[sp],
            'real_score': real,
            'null_mean': float(np.mean(null)) if len(null) else np.nan,
            'null_std': float(np.std(null)) if len(null) else np.nan,
            'p_value': p,
        })
    df = pd.DataFrame(rows)
    valid = df['p_value'].notna()
    q = np.full(len(df), np.nan)
    if valid.any():
        q[valid.values] = bh_fdr(df.loc[valid, 'p_value'].values)
    df['q_value'] = q
    df['significant_fdr'] = df['q_value'] < args.fdr_alpha
    df = df.sort_values('p_value')
    df.to_csv(outdir / 'permutation_pvalues_pathway.csv', index=False)

    n_sig = int(df['significant_fdr'].sum())
    logger.info(
        f'Significant sub-pathways (FDR<{args.fdr_alpha}): {n_sig}/{len(df)}'
    )
    logger.info(f'Top 10 by p-value:')
    for _, r in df.head(10).iterrows():
        logger.info(
            f"  {r['sub_pathway']:40s} n={r['n_metabolites']:3d} "
            f"p={r['p_value']:.4f} q={r['q_value']:.4f}"
        )
    logger.info(f'Results: {outdir / "permutation_pvalues_pathway.csv"}')


if __name__ == '__main__':
    main()
