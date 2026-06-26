#!/usr/bin/env python3
"""
Per-feature permutation test for the Pain HeteroGNN pipeline.

For each metabolite and enzyme, compute an empirical null distribution of
importance scores by re-running 6-fold CV on label-shuffled data, then
compute p-values and BH-FDR q-values.

Usage:
    python permutation_test_features.py \
        --csv starting_template.csv \
        --outdir pain_hetero_pipeline/out_hetero_endogenous_6fold \
        --n-perms 200 --seed 42

Outputs (into --outdir):
    permutation_null_met.csv      : metabolite x permutation null scores
    permutation_null_enz.csv      : enzyme x permutation null scores
    permutation_pvalues_met.csv   : metabolite, real_score, p, q, significant
    permutation_pvalues_enz.csv   : enzyme, real_score, p, q, significant
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

# Make sibling pipeline modules importable
_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from pain_hetero_pipeline import (
    PainHeteroGNN,
    build_hetero_graph,
    train_epoch,
    evaluate,
    compute_metabolite_importance,
    compute_enzyme_scores,
)
from utils import load_metabolomics_data, preprocess_fold

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('perm_test')


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_one_cv(X, y, metadata, met_cols, met2ec, met2pathway, args, device):
    """Run one full 6-fold CV and return aggregated metabolite + enzyme scores."""
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True,
                          random_state=args.seed)

    met_score_accum = defaultdict(list)
    enz_score_accum = defaultdict(list)
    fold_aucs = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        age = metadata.get('age')
        sev = metadata.get('severity')
        age_train = age[train_idx] if age is not None else None
        age_val = age[val_idx] if age is not None else None
        sev_train = sev[train_idx] if sev is not None else None
        sev_val = sev[val_idx] if sev is not None else None

        X_train_s, X_val_s, sf_train, sf_val = preprocess_fold(
            X_train, X_val, age_train, age_val, sev_train, sev_val
        )
        X_all = np.vstack([X_train_s, X_val_s])
        y_all = np.concatenate([y_train, y_val])
        sf_all = np.vstack([sf_train, sf_val])

        n_train = len(train_idx)
        n_total = len(X_all)
        train_mask = np.zeros(n_total, dtype=bool)
        val_mask = np.zeros(n_total, dtype=bool)
        train_mask[:n_train] = True
        val_mask[n_train:] = True

        data = build_hetero_graph(
            X_all, y_all, sf_all, met_cols,
            met2ec, met2pathway,
            train_mask, val_mask,
            use_metmet_prior=args.use_metmet_prior,
            k_metmet=args.k_metmet,
            seed=args.seed,
        )

        model = PainHeteroGNN(
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
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        best_auc = 0.0
        best_state = None
        patience = 0
        for epoch in range(args.epochs):
            train_epoch(model, data, optimizer, criterion, device)
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
        fold_aucs.append(auc)

        met_imp = compute_metabolite_importance(model, data, device)
        enz_scores = compute_enzyme_scores(met_imp, met2ec)

        for m, s in met_imp.items():
            met_score_accum[m].append(s)
        for ec, info in enz_scores.items():
            enz_score_accum[ec].append(info['score'])

    met_mean = {m: float(np.mean(v)) for m, v in met_score_accum.items()}
    enz_mean = {e: float(np.mean(v)) for e, v in enz_score_accum.items()}

    # Normalize to proportions so runs are comparable regardless of scale
    met_total = sum(met_mean.values())
    if met_total > 0:
        met_mean = {m: v / met_total for m, v in met_mean.items()}
    enz_total = sum(enz_mean.values())
    if enz_total > 0:
        enz_mean = {e: v / enz_total for e, v in enz_mean.items()}

    return met_mean, enz_mean, float(np.mean(fold_aucs))


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns q-values aligned to input."""
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * n / (np.arange(n) + 1)
    # Enforce monotonicity from the right
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(n)
    out[order] = q
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--n-perms', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-folds', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--high-threshold', type=int, default=4)
    parser.add_argument('--low-threshold', type=int, default=3)
    parser.add_argument('--exclude-xenobiotics', action='store_true')
    parser.add_argument('--annotation-path', type=str,
                        default='metabolites names and pathways.xlsx')
    parser.add_argument('--use-metmet-prior', action='store_true')
    parser.add_argument('--k-metmet', type=int, default=5)
    parser.add_argument('--fdr-alpha', type=float, default=0.05)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    # Load data
    met_df, labels, group_labels, met_cols, metadata = load_metabolomics_data(
        args.csv,
        high_threshold=args.high_threshold,
        low_threshold=args.low_threshold,
        exclude_xenobiotics=args.exclude_xenobiotics,
        annotation_path=args.annotation_path,
    )
    X = met_df.values.astype(np.float32)
    y = np.asarray(labels)
    logger.info(f'Loaded: X={X.shape}, y={y.shape}, mets={len(met_cols)}')

    # Reuse existing KEGG mappings if present
    met2ec_path = outdir / 'met2ec.json'
    met2pw_path = outdir / 'met2pathway.json'
    if not (met2ec_path.exists() and met2pw_path.exists()):
        raise RuntimeError(
            f'Missing {met2ec_path} / {met2pw_path}. Run the main pipeline '
            f'first so KEGG mappings are cached.'
        )
    with open(met2ec_path) as f:
        met2ec = json.load(f)
    with open(met2pw_path) as f:
        met2pathway = json.load(f)
    logger.info(f'Loaded KEGG maps: {len(met2ec)} met2ec entries')

    # Load real importance from the main pipeline run (for real scores)
    real_met_df = pd.read_csv(outdir / 'metabolite_rank.csv')
    real_enz_df = pd.read_csv(outdir / 'enzyme_rank.csv')
    real_met_score_raw = dict(zip(real_met_df['metabolite'],
                                   real_met_df['importance_score']))
    real_enz_score_raw = dict(zip(real_enz_df['enzyme_id'],
                                   real_enz_df['enzyme_score']))

    # Normalize real scores to proportions (same as permutation runs)
    met_total = sum(real_met_score_raw.values())
    real_met_score = {m: v / met_total for m, v in real_met_score_raw.items()} if met_total > 0 else real_met_score_raw
    enz_total = sum(real_enz_score_raw.values())
    real_enz_score = {e: v / enz_total for e, v in real_enz_score_raw.items()} if enz_total > 0 else real_enz_score_raw

    logger.info(
        f'Real scores: {len(real_met_score)} mets, {len(real_enz_score)} enzymes (normalized to proportions)'
    )

    # Build null distributions
    set_all_seeds(args.seed)
    met_null = defaultdict(list)
    enz_null = defaultdict(list)
    perm_aucs = []

    t0 = time.time()
    for perm_idx in range(args.n_perms):
        y_perm = np.random.permutation(y)
        try:
            m_mean, e_mean, auc = run_one_cv(
                X, y_perm, metadata, met_cols, met2ec, met2pathway,
                args, device
            )
        except Exception as exc:
            logger.warning(f'perm {perm_idx} failed: {exc}')
            continue

        perm_aucs.append(auc)
        for m, s in m_mean.items():
            met_null[m].append(s)
        for e, s in e_mean.items():
            enz_null[e].append(s)

        elapsed = time.time() - t0
        if (perm_idx + 1) % 5 == 0 or perm_idx == 0:
            eta = elapsed / (perm_idx + 1) * (args.n_perms - perm_idx - 1)
            logger.info(
                f'perm {perm_idx + 1}/{args.n_perms} | auc={auc:.3f} '
                f'| elapsed={elapsed/60:.1f}m | ETA={eta/60:.1f}m'
            )

    logger.info(
        f'Perm AUC: mean={np.mean(perm_aucs):.3f}, std={np.std(perm_aucs):.3f}'
    )

    # ---- Save raw null distributions ----
    met_null_rows = []
    for m, scores in met_null.items():
        row = {'metabolite': m, 'n_perms': len(scores),
               'null_mean': float(np.mean(scores)),
               'null_std': float(np.std(scores)),
               'null_p95': float(np.percentile(scores, 95)),
               'null_max': float(np.max(scores))}
        met_null_rows.append(row)
    pd.DataFrame(met_null_rows).to_csv(
        outdir / 'permutation_null_met.csv', index=False
    )

    enz_null_rows = []
    for e, scores in enz_null.items():
        row = {'enzyme_id': e, 'n_perms': len(scores),
               'null_mean': float(np.mean(scores)),
               'null_std': float(np.std(scores)),
               'null_p95': float(np.percentile(scores, 95)),
               'null_max': float(np.max(scores))}
        enz_null_rows.append(row)
    pd.DataFrame(enz_null_rows).to_csv(
        outdir / 'permutation_null_enz.csv', index=False
    )

    # ---- Compute p-values ----
    def compute_p(real_map, null_map):
        feats = sorted(real_map.keys())
        rows = []
        for f in feats:
            real = real_map[f]
            null_scores = np.array(null_map.get(f, []))
            if len(null_scores) == 0:
                p = np.nan
            else:
                # one-sided: is real score larger than null?
                p = (1 + int((null_scores >= real).sum())) / (len(null_scores) + 1)
            rows.append({'feature': f, 'real_score': real,
                         'n_null': len(null_scores), 'p_value': p})
        df = pd.DataFrame(rows)
        valid = df['p_value'].notna()
        q = np.full(len(df), np.nan)
        if valid.any():
            q[valid.values] = bh_fdr(df.loc[valid, 'p_value'].values)
        df['q_value'] = q
        df['significant_fdr'] = df['q_value'] < args.fdr_alpha
        return df.sort_values('p_value')

    met_p = compute_p(real_met_score, met_null)
    met_p = met_p.rename(columns={'feature': 'metabolite'})
    met_p.to_csv(outdir / 'permutation_pvalues_met.csv', index=False)

    enz_p = compute_p(real_enz_score, enz_null)
    enz_p = enz_p.rename(columns={'feature': 'enzyme_id'})
    enz_p.to_csv(outdir / 'permutation_pvalues_enz.csv', index=False)

    n_sig_met = int(met_p['significant_fdr'].sum())
    n_sig_enz = int(enz_p['significant_fdr'].sum())
    logger.info(
        f'Significant (FDR<{args.fdr_alpha}): mets={n_sig_met}/{len(met_p)}, '
        f'enzymes={n_sig_enz}/{len(enz_p)}'
    )
    logger.info(f'Results written to {outdir}')


if __name__ == '__main__':
    main()
