#!/usr/bin/env python3
"""
XAI Tier 3: Confounder analysis.

Runs the SAME GNN pipeline but with DIFFERENT target labels to check
if the Pain signal overlaps with confounders:
  1. Pain (High vs Low) - the main model (reference)
  2. Group (HZ_HL vs PHN)
  3. Severity (Severe vs non-Severe)
  4. Age (old >= median vs young)

Compares top-30 enzyme lists across these targets to reveal which
enzymes are pain-specific vs confounded with another covariate.
"""

import os
import sys
import json
import argparse
import logging
import random
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
    compute_metabolite_importance, compute_enzyme_scores,
)
from utils import load_metabolomics_data, preprocess_fold

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('xai_conf')


def set_all_seeds(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def run_full(X, y_target, metadata, met_cols, met2ec, met2pathway,
             args, device):
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True,
                          random_state=args.seed)
    enz_accum = defaultdict(list)
    aucs = []
    for tr, vl in skf.split(X, y_target):
        X_tr, X_vl = X[tr], X[vl]
        y_tr, y_vl = y_target[tr], y_target[vl]
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
                    if patience >= args.patience: break
        if best_state is not None:
            model.load_state_dict(best_state)
        _, auc, *_ = evaluate(model, data, device)
        aucs.append(auc)
        met_imp = compute_metabolite_importance(model, data, device)
        enz_sc = compute_enzyme_scores(met_imp, met2ec)
        for ec, info in enz_sc.items():
            enz_accum[ec].append(info['score'])
    enz_mean = {e: float(np.mean(v)) for e, v in enz_accum.items()}
    return enz_mean, float(np.mean(aucs)), float(np.std(aucs))


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
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    met_df, y_pain, group_labels, met_cols, metadata = load_metabolomics_data(
        args.csv, high_threshold=args.high_threshold,
        low_threshold=args.low_threshold,
        exclude_xenobiotics=args.exclude_xenobiotics,
        annotation_path=args.annotation_path,
    )
    X = met_df.values.astype(np.float32)
    y_pain = np.asarray(y_pain)
    with open(outdir / 'met2ec.json') as f: met2ec = json.load(f)
    with open(outdir / 'met2pathway.json') as f: met2pathway = json.load(f)

    # Build target labels
    targets = {'pain_HighVsLow': y_pain.astype(int)}

    # Group: PHN (1) vs HZ_HL (0)
    group_arr = np.array([1 if g == 'PHN' else 0 for g in group_labels])
    targets['group_PHNvsHZHL'] = group_arr

    # Severity: Severe (1) vs non-Severe (0); drop 'Unknown'
    sev = metadata.get('severity')
    if sev is not None:
        sev_bin = np.array([1 if str(s).strip().lower() == 'severe' else 0
                            for s in sev])
        targets['severity_SevereVsOther'] = sev_bin

    # Age: above median (1) vs below
    age = metadata.get('age')
    if age is not None:
        age_arr = np.asarray(age, dtype=float)
        med = float(np.median(age_arr))
        targets['age_oldVsYoung'] = (age_arr >= med).astype(int)

    # Run GNN for each target
    all_results = {}
    summary_rows = []
    for name, y_tgt in targets.items():
        pos = int(y_tgt.sum()); neg = int(len(y_tgt) - pos)
        if pos < 3 or neg < 3:
            logger.info(f'{name}: skip (pos={pos}, neg={neg}, too imbalanced)')
            continue
        set_all_seeds(args.seed)
        logger.info(f'Running target="{name}" (pos={pos}, neg={neg})...')
        try:
            enz_mean, auc_m, auc_s = run_full(
                X, y_tgt, metadata, met_cols, met2ec, met2pathway,
                args, device
            )
        except Exception as e:
            logger.error(f'{name} failed: {e}')
            continue
        all_results[name] = enz_mean
        summary_rows.append({'target': name, 'n_pos': pos, 'n_neg': neg,
                             'auc_mean': auc_m, 'auc_std': auc_s})
        logger.info(f'  AUC={auc_m:.3f}±{auc_s:.3f}')

    # Save per-target top 30
    all_top30 = {}
    rows = []
    for name, enz_mean in all_results.items():
        top30 = sorted(enz_mean.items(), key=lambda x: -x[1])[:30]
        all_top30[name] = [e for e, _ in top30]
        for rank, (ec, s) in enumerate(top30, 1):
            rows.append({'target': name, 'rank': rank,
                         'enzyme_id': ec, 'score': s})
    pd.DataFrame(rows).to_csv(outdir / 'xai_confounders_top30.csv',
                              index=False)

    # Overlap matrix
    names = list(all_top30.keys())
    overlap_rows = []
    for n1 in names:
        for n2 in names:
            s1 = set(all_top30[n1])
            s2 = set(all_top30[n2])
            ov = len(s1 & s2)
            overlap_rows.append({'target_A': n1, 'target_B': n2,
                                 'top30_overlap': ov})
    ov_df = pd.DataFrame(overlap_rows)
    piv = ov_df.pivot(index='target_A', columns='target_B',
                      values='top30_overlap')
    piv.to_csv(outdir / 'xai_confounders_overlap.csv')
    logger.info('\n=== Top-30 overlap matrix ===')
    logger.info(piv.to_string())

    pd.DataFrame(summary_rows).to_csv(
        outdir / 'xai_confounders_summary.csv', index=False
    )

    # Pain-specific enzymes: in pain top-30 but NOT in any confounder top-30
    pain_top = set(all_top30.get('pain_HighVsLow', []))
    confounders_union = set()
    for nm, lst in all_top30.items():
        if nm != 'pain_HighVsLow':
            confounders_union |= set(lst)
    pain_specific = sorted(pain_top - confounders_union)
    pain_shared = sorted(pain_top & confounders_union)
    pd.DataFrame({'enzyme_id': list(pain_specific),
                  'category': ['pain_specific'] * len(pain_specific)}
                 ).to_csv(outdir / 'xai_pain_specific_enzymes.csv', index=False)
    pd.DataFrame({'enzyme_id': list(pain_shared),
                  'category': ['shared_with_confounder'] * len(pain_shared)}
                 ).to_csv(outdir / 'xai_pain_shared_enzymes.csv', index=False)
    logger.info(
        f'\nPain top-30: {len(pain_specific)} unique to pain, '
        f'{len(pain_shared)} shared with at least one confounder'
    )
    logger.info(f'Pain-specific enzymes: {pain_specific}')
    logger.info(f'Shared-with-confounder: {pain_shared}')


if __name__ == '__main__':
    main()
