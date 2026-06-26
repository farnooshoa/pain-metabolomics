#!/usr/bin/env python3
"""
XAI Tier 1, Part B: Ablation study.

Remove top-K metabolites (by GNN importance) from the input, retrain GNN,
measure AUC drop. Also removes matched random-K sets as control.

Usage:
    python xai_ablation.py --csv starting_template.csv \
        --outdir pain_hetero_pipeline/out_hetero_endogenous_6fold \
        --k-list 0,10,30,50,100,200 --n-random 5
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
sys.path.insert(0, str(_here))

from pain_hetero_pipeline import (
    PainHeteroGNN, build_hetero_graph, train_epoch, evaluate,
)
from utils import load_metabolomics_data, preprocess_fold

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('xai_abl')


def set_all_seeds(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def run_cv(X, y, metadata, met_cols, met2ec, met2pathway, args, device, seed):
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True,
                          random_state=seed)
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
    return float(np.mean(aucs)), float(np.std(aucs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--outdir', required=True)
    p.add_argument('--k-list', type=str, default='0,10,30,50,100,200')
    p.add_argument('--n-random', type=int, default=5,
                   help='Number of random-K control runs per K')
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
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    # Load
    met_df, y, _, met_cols, metadata = load_metabolomics_data(
        args.csv, high_threshold=args.high_threshold,
        low_threshold=args.low_threshold,
        exclude_xenobiotics=args.exclude_xenobiotics,
        annotation_path=args.annotation_path,
    )
    X = met_df.values.astype(np.float32)
    y = np.asarray(y)
    with open(outdir / 'met2ec.json') as f: met2ec = json.load(f)
    with open(outdir / 'met2pathway.json') as f: met2pathway = json.load(f)

    # Load GNN top-ranked metabolites
    met_rank = pd.read_csv(outdir / 'metabolite_rank.csv')
    ranked_mets = met_rank['metabolite'].tolist()

    k_list = [int(k) for k in args.k_list.split(',')]
    rows = []
    t0 = time.time()

    for k in k_list:
        # ---- Remove top K ----
        if k == 0:
            remove = set()
            label = 'top0 (baseline)'
        else:
            remove = set(ranked_mets[:k])
            label = f'top{k}'
        keep_idx = [i for i, m in enumerate(met_cols) if m not in remove]
        if len(keep_idx) < 10:
            logger.warning(f'K={k}: too few metabolites left ({len(keep_idx)})')
            continue
        X_sub = X[:, keep_idx]
        mc_sub = [met_cols[i] for i in keep_idx]
        set_all_seeds(args.seed)
        auc_m, auc_s = run_cv(
            X_sub, y, metadata, mc_sub, met2ec, met2pathway,
            args, device, seed=args.seed
        )
        elapsed = (time.time() - t0) / 60
        logger.info(
            f'K={k:3d} [remove-top]    AUC={auc_m:.3f}±{auc_s:.3f}  '
            f'(n_mets={len(mc_sub)}) elapsed={elapsed:.1f}m'
        )
        rows.append({'k': k, 'type': 'remove_top', 'rep': 0,
                     'auc_mean': auc_m, 'auc_std': auc_s,
                     'n_mets_remaining': len(mc_sub)})

        # ---- Remove random K as control ----
        if k > 0:
            for rep in range(args.n_random):
                rng = np.random.default_rng(1000 + rep + k)
                rand_idx = rng.choice(len(met_cols), k, replace=False)
                rand_set = set(met_cols[i] for i in rand_idx)
                keep_idx_r = [i for i, m in enumerate(met_cols) if m not in rand_set]
                X_sub_r = X[:, keep_idx_r]
                mc_sub_r = [met_cols[i] for i in keep_idx_r]
                set_all_seeds(args.seed + rep)
                auc_mr, auc_sr = run_cv(
                    X_sub_r, y, metadata, mc_sub_r, met2ec, met2pathway,
                    args, device, seed=args.seed + rep
                )
                rows.append({'k': k, 'type': 'remove_random', 'rep': rep,
                             'auc_mean': auc_mr, 'auc_std': auc_sr,
                             'n_mets_remaining': len(mc_sub_r)})
            # Report mean random
            rand_aucs = [r['auc_mean'] for r in rows
                         if r['k'] == k and r['type'] == 'remove_random']
            elapsed = (time.time() - t0) / 60
            logger.info(
                f'K={k:3d} [remove-random]  AUC={np.mean(rand_aucs):.3f}'
                f'±{np.std(rand_aucs):.3f} (over {len(rand_aucs)} reps) '
                f'elapsed={elapsed:.1f}m'
            )

    df = pd.DataFrame(rows)
    df.to_csv(outdir / 'xai_ablation_results.csv', index=False)

    # Summary
    summary_rows = []
    baseline_auc = df[df['k'] == 0]['auc_mean'].values[0] if 0 in k_list else np.nan
    for k in sorted(df['k'].unique()):
        top_auc = df[(df['k'] == k) & (df['type'] == 'remove_top')]['auc_mean'].values
        rand_aucs = df[(df['k'] == k) & (df['type'] == 'remove_random')]['auc_mean'].values
        summary_rows.append({
            'k': k,
            'auc_remove_top': float(top_auc[0]) if len(top_auc) else np.nan,
            'auc_remove_random_mean': float(np.mean(rand_aucs)) if len(rand_aucs) else np.nan,
            'auc_remove_random_std': float(np.std(rand_aucs)) if len(rand_aucs) else np.nan,
            'auc_delta_vs_baseline': float(top_auc[0] - baseline_auc) if len(top_auc) else np.nan,
        })
    sdf = pd.DataFrame(summary_rows)
    sdf.to_csv(outdir / 'xai_ablation_summary.csv', index=False)
    logger.info('\n=== ABLATION SUMMARY ===')
    logger.info(sdf.to_string())
    logger.info(f'All results: {outdir}')


if __name__ == '__main__':
    main()
