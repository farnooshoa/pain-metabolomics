#!/usr/bin/env python3
"""
Subsample power analysis: downsample the current n=36 cohort to n=12, 18,
24, 30 and measure how AUC stability (std across repeated runs) changes.
Extrapolate upward to estimate the sample size needed for stable AUC.
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
)
from utils import load_metabolomics_data, preprocess_fold

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('power')


def run_cv(X, y, metadata, met_cols, met2ec, met2pathway, args, device, seed):
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=seed)
    aucs = []
    for fold_idx, (tr_idx, vl_idx) in enumerate(skf.split(X, y)):
        X_tr, X_vl = X[tr_idx], X[vl_idx]
        y_tr, y_vl = y[tr_idx], y[vl_idx]
        age = metadata.get('age')
        sev = metadata.get('severity')
        age_t = age[tr_idx] if age is not None else None
        age_v = age[vl_idx] if age is not None else None
        sev_t = sev[tr_idx] if sev is not None else None
        sev_v = sev[vl_idx] if sev is not None else None
        X_ts, X_vs, sf_t, sf_v = preprocess_fold(X_tr, X_vl, age_t, age_v, sev_t, sev_v)
        X_all = np.vstack([X_ts, X_vs])
        y_all = np.concatenate([y_tr, y_vl])
        sf_all = np.vstack([sf_t, sf_v])
        n_tr = len(tr_idx)
        tr_mask = np.zeros(len(X_all), dtype=bool); tr_mask[:n_tr] = True
        vl_mask = np.zeros(len(X_all), dtype=bool); vl_mask[n_tr:] = True
        data = build_hetero_graph(
            X_all, y_all, sf_all, met_cols, met2ec, met2pathway,
            tr_mask, vl_mask,
            use_metmet_prior=args.use_metmet_prior,
            k_metmet=args.k_metmet, seed=seed,
        )
        model = PainHeteroGNN(
            sample_in_dim=sf_all.shape[1], hidden_dim=args.hidden_dim,
            n_layers=args.n_layers, dropout=args.dropout,
            use_metmet_prior=args.use_metmet_prior,
        ).to(device)
        pos_w = torch.tensor([(len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)]).to(device)
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
    return float(np.mean(aucs))


def stratified_subsample(y, n_target, seed):
    rng = np.random.default_rng(seed)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    ratio_pos = len(idx_pos) / len(y)
    n_pos = max(2, int(round(n_target * ratio_pos)))
    n_neg = max(2, n_target - n_pos)
    n_pos = min(n_pos, len(idx_pos))
    n_neg = min(n_neg, len(idx_neg))
    sel_pos = rng.choice(idx_pos, n_pos, replace=False)
    sel_neg = rng.choice(idx_neg, n_neg, replace=False)
    return np.concatenate([sel_pos, sel_neg])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--input-dir', required=True)
    p.add_argument('--outdir', required=True)
    p.add_argument('--sizes', type=str, default='12,18,24,30,36')
    p.add_argument('--n-reps', type=int, default=10)
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
    args = p.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    input_dir = Path(args.input_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sizes = [int(s) for s in args.sizes.split(',')]

    met_df, labels, _, met_cols, metadata = load_metabolomics_data(
        args.csv, high_threshold=args.high_threshold,
        low_threshold=args.low_threshold,
        exclude_xenobiotics=args.exclude_xenobiotics,
        annotation_path=args.annotation_path,
    )
    X = met_df.values.astype(np.float32)
    y = np.asarray(labels)
    with open(input_dir / 'met2ec.json') as f: met2ec = json.load(f)
    with open(input_dir / 'met2pathway.json') as f: met2pathway = json.load(f)

    # Need per-rep X,y,metadata slices
    results = []
    for n_target in sizes:
        n_folds_eff = min(args.n_folds, n_target // 2)
        args_copy = argparse.Namespace(**vars(args))
        args_copy.n_folds = n_folds_eff
        for rep in range(args.n_reps):
            if n_target >= len(y):
                idx = np.arange(len(y))
                if rep > 0:
                    continue  # no subsampling needed, only 1 rep at full size
            else:
                idx = stratified_subsample(y, n_target, seed=1000 + rep)
            X_sub = X[idx]; y_sub = y[idx]
            metadata_sub = {
                'age': metadata['age'][idx] if metadata.get('age') is not None else None,
                'severity': metadata['severity'][idx] if metadata.get('severity') is not None else None,
            }
            try:
                auc = run_cv(X_sub, y_sub, metadata_sub, met_cols,
                             met2ec, met2pathway, args_copy, device,
                             seed=2000 + rep)
            except Exception as e:
                logger.warning(f'n={n_target} rep={rep} failed: {e}')
                continue
            results.append({'n': n_target, 'rep': rep, 'auc': auc,
                            'n_pos': int(y_sub.sum()),
                            'n_neg': int(len(y_sub) - y_sub.sum())})
            logger.info(f'n={n_target} rep={rep+1}/{args.n_reps} AUC={auc:.3f}')

    df = pd.DataFrame(results)
    df.to_csv(outdir / 'power_analysis_subsample.csv', index=False)

    summary = df.groupby('n').agg(
        auc_mean=('auc', 'mean'),
        auc_std=('auc', 'std'),
        n_reps=('auc', 'count'),
    ).reset_index()
    summary.to_csv(outdir / 'power_analysis_summary.csv', index=False)
    logger.info('Summary:')
    logger.info(summary.to_string())


if __name__ == '__main__':
    main()
