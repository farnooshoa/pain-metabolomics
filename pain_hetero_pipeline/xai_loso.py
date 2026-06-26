#!/usr/bin/env python3
"""
XAI Tier 1, Part C: Leave-One-Sample-Out sensitivity.

For each of the 36 samples: remove it, rerun the full 6-fold CV on the
remaining 35, save the top-30 enzyme list + AUC. Then compute:
  - AUC range / mean across 36 LOSO runs
  - Enzyme churn rate: how often does each Top-30 enzyme stay in Top-30?
  - Sample-influence score: how much does removing sample X change the rankings?
"""

import os
import sys
import json
import argparse
import logging
import random
import time
from pathlib import Path
from collections import defaultdict, Counter
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))

from pain_hetero_pipeline import (
    PainHeteroGNN, build_hetero_graph, train_epoch, evaluate,
    compute_metabolite_importance, compute_enzyme_scores,
)
from utils import load_metabolomics_data, preprocess_fold

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('xai_loso')


def set_all_seeds(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def run_full_cv(X, y, metadata, met_cols, met2ec, met2pathway,
                args, device):
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True,
                          random_state=args.seed)
    met_accum = defaultdict(list)
    enz_accum = defaultdict(list)
    aucs = []
    for tr, vl in skf.split(X, y):
        X_tr, X_vl = X[tr], X[vl]
        y_tr, y_vl = y[tr], y[vl]
        age = metadata.get('age')
        sev = metadata.get('severity')
        age_t = age[tr] if age is not None else None
        age_v = age[vl] if age is not None else None
        sev_t = sev[tr] if sev is not None else None
        sev_v = sev[vl] if sev is not None else None
        X_ts, X_vs, sf_t, sf_v = preprocess_fold(
            X_tr, X_vl, age_t, age_v, sev_t, sev_v
        )
        X_all = np.vstack([X_ts, X_vs])
        y_all = np.concatenate([y_tr, y_vl])
        sf_all = np.vstack([sf_t, sf_v])
        n_tr_ = len(tr)
        tm = np.zeros(len(X_all), dtype=bool); tm[:n_tr_] = True
        vm = np.zeros(len(X_all), dtype=bool); vm[n_tr_:] = True
        data = build_hetero_graph(
            X_all, y_all, sf_all, met_cols, met2ec, met2pathway,
            tm, vm, use_metmet_prior=args.use_metmet_prior,
            k_metmet=args.k_metmet, seed=args.seed,
        )
        model = PainHeteroGNN(
            sample_in_dim=sf_all.shape[1], hidden_dim=args.hidden_dim,
            n_layers=args.n_layers, dropout=args.dropout,
            use_metmet_prior=args.use_metmet_prior,
        ).to(device)
        pos_w = torch.tensor(
            [(len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)]
        ).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
        best_auc = 0.0; best_state = None; patience = 0
        for epoch in range(args.epochs):
            train_epoch(model, data, opt, criterion, device)
            if (epoch + 1) % 10 == 0:
                _, auc, *_ = evaluate(model, data, device)
                if auc > best_auc:
                    best_auc = auc; best_state = deepcopy(model.state_dict())
                    patience = 0
                else:
                    patience += 1
                    if patience >= args.patience:
                        break
        if best_state is not None:
            model.load_state_dict(best_state)
        _, auc, *_ = evaluate(model, data, device)
        aucs.append(auc)
        met_imp = compute_metabolite_importance(model, data, device)
        enz_sc = compute_enzyme_scores(met_imp, met2ec)
        for m, s in met_imp.items():
            met_accum[m].append(s)
        for ec, info in enz_sc.items():
            enz_accum[ec].append(info['score'])
    met_mean = {m: float(np.mean(v)) for m, v in met_accum.items()}
    enz_mean = {e: float(np.mean(v)) for e, v in enz_accum.items()}
    return met_mean, enz_mean, float(np.mean(aucs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--outdir', required=True)
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
    p.add_argument('--sample-ids-col', type=str, default='CLIENT_SAMPLE_ID')
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    met_df, y, _, met_cols, metadata = load_metabolomics_data(
        args.csv, high_threshold=args.high_threshold,
        low_threshold=args.low_threshold,
        exclude_xenobiotics=args.exclude_xenobiotics,
        annotation_path=args.annotation_path,
    )
    X = met_df.values.astype(np.float32)
    y = np.asarray(y)
    sample_ids = metadata.get('sample_ids', np.arange(len(y)))
    with open(outdir / 'met2ec.json') as f: met2ec = json.load(f)
    with open(outdir / 'met2pathway.json') as f: met2pathway = json.load(f)

    # Full-data top 30 enzymes (reference)
    ref_enz = pd.read_csv(outdir / 'enzyme_rank.csv').head(30)['enzyme_id'].tolist()
    ref_set = set(ref_enz)

    # LOSO loop
    n = len(y)
    loso_rows = []
    enz_membership_counts = Counter()  # how often each ref enzyme stays in top30
    new_enz_intrusions = Counter()     # new enzymes appearing in top30

    t0 = time.time()
    for i in range(n):
        keep = np.ones(n, dtype=bool); keep[i] = False
        X_sub = X[keep]
        y_sub = y[keep]
        metadata_sub = {
            'age': metadata['age'][keep] if metadata.get('age') is not None else None,
            'severity': metadata['severity'][keep] if metadata.get('severity') is not None else None,
        }
        try:
            met_mean, enz_mean, auc = run_full_cv(
                X_sub, y_sub, metadata_sub, met_cols,
                met2ec, met2pathway, args, device
            )
        except Exception as e:
            logger.warning(f'LOSO {i} ({sample_ids[i]}) failed: {e}')
            continue

        top30 = sorted(enz_mean.items(), key=lambda x: -x[1])[:30]
        top30_set = set(e for e, _ in top30)
        overlap = len(top30_set & ref_set)
        for ec in top30_set & ref_set:
            enz_membership_counts[ec] += 1
        for ec in top30_set - ref_set:
            new_enz_intrusions[ec] += 1

        loso_rows.append({
            'removed_sample_idx': i,
            'removed_sample_id': str(sample_ids[i]),
            'removed_label': int(y[i]),
            'auc': auc,
            'top30_overlap_with_full': overlap,
            'top30_churn': 30 - overlap,
        })
        el = (time.time() - t0) / 60
        eta = el / (i + 1) * (n - i - 1)
        logger.info(
            f'LOSO {i + 1}/{n} ({sample_ids[i]}, label={y[i]}): '
            f'AUC={auc:.3f}, top30 overlap={overlap}/30 | '
            f'elapsed={el:.1f}m, ETA={eta:.1f}m'
        )

    df = pd.DataFrame(loso_rows)
    df.to_csv(outdir / 'xai_loso_results.csv', index=False)

    # Per-enzyme stability: how often does each ref top-30 enzyme stay?
    stab_rows = []
    for ec in ref_enz:
        stab_rows.append({
            'enzyme_id': ec,
            'n_loso_runs': len(df),
            'n_times_in_top30': int(enz_membership_counts[ec]),
            'stability_rate': enz_membership_counts[ec] / len(df) if len(df) else 0,
        })
    stab_df = pd.DataFrame(stab_rows).sort_values('stability_rate', ascending=False)
    stab_df.to_csv(outdir / 'xai_loso_enzyme_stability.csv', index=False)

    logger.info('\n=== LOSO SUMMARY ===')
    logger.info(f'AUC: mean={df["auc"].mean():.3f}, '
                f'std={df["auc"].std():.3f}, '
                f'min={df["auc"].min():.3f}, max={df["auc"].max():.3f}')
    logger.info(f'Top-30 churn (enzymes removed from ref top-30): '
                f'mean={df["top30_churn"].mean():.1f}, '
                f'max={df["top30_churn"].max()} (worst single sample)')
    logger.info(f'Most influential samples (largest churn):')
    worst = df.nlargest(5, 'top30_churn')[['removed_sample_id', 'removed_label',
                                            'auc', 'top30_churn']]
    logger.info(worst.to_string())
    logger.info('\nReference Top-30 enzyme stability (how often they stay in top-30):')
    logger.info(stab_df.head(15).to_string())


if __name__ == '__main__':
    main()
