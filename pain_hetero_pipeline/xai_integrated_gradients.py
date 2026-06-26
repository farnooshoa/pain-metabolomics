#!/usr/bin/env python3
"""
XAI Tier 2: Integrated Gradients + per-sample attribution.

For each sample, compute Integrated Gradients over the sample-met edge
attributes (abundances) relative to a per-metabolite baseline (mean of
training abundances). This yields a signed per-feature attribution
whose sum equals f(x) - f(baseline), giving principled per-patient
explanations.

Outputs:
    xai_ig_per_sample.csv          : sample x metabolite IG attribution matrix
    xai_ig_per_sample_top10.csv    : top 10 features pushing each sample
                                     toward High/Low pain
    xai_ig_global_importance.csv   : mean |IG| per metabolite (global view)
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
)
from utils import load_metabolomics_data, preprocess_fold

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('xai_ig')


def set_all_seeds(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def integrated_gradients_per_sample(model, data, device, n_steps=50,
                                    sample_indices=None):
    """Compute IG attribution per sample over sample-met edge_attr.

    Returns dict: sample_idx -> np.ndarray [n_mets]
    Also returns global importance = mean |IG| across samples.
    """
    model.eval()
    data = data.to(device)
    orig_edge_attr = data['sample', 'has', 'met'].edge_attr.detach().clone()
    edge_index = data['sample', 'has', 'met'].edge_index.cpu().numpy()

    n_samples = data['sample'].x.shape[0]
    n_mets = len(data.metabolite_names)
    if sample_indices is None:
        sample_indices = list(range(n_samples))

    # Baseline: for each edge (sample_i, metabolite_j), use the mean abundance
    # of metabolite j across all samples (an "average patient" baseline).
    abundances = orig_edge_attr.squeeze(-1) if orig_edge_attr.dim() > 1 else orig_edge_attr
    sample_idx_arr = edge_index[0]
    met_idx_arr = edge_index[1]

    # mean per metabolite (across edges to that metabolite, which == across samples that have it)
    met_mean = np.zeros(n_mets, dtype=np.float32)
    met_count = np.zeros(n_mets, dtype=np.int64)
    abund_np = abundances.cpu().numpy()
    for i, mj in enumerate(met_idx_arr):
        met_mean[mj] += float(abund_np[i])
        met_count[mj] += 1
    met_count[met_count == 0] = 1
    met_mean = met_mean / met_count

    # Edge-level baseline: use met_mean of the metabolite each edge connects to
    baseline_edge = torch.tensor(met_mean[met_idx_arr],
                                 device=device, dtype=torch.float32)
    if orig_edge_attr.dim() > 1:
        baseline_edge = baseline_edge.unsqueeze(-1)

    per_sample_ig = np.zeros((len(sample_indices), n_mets), dtype=np.float32)

    for si, s_idx in enumerate(sample_indices):
        # Accumulate gradient along integration path
        grad_accum = torch.zeros_like(orig_edge_attr)
        for k in range(1, n_steps + 1):
            alpha = k / n_steps
            interp = baseline_edge + alpha * (orig_edge_attr - baseline_edge)
            ea = interp.clone().requires_grad_(True)
            data['sample', 'has', 'met'].edge_attr = ea
            logits = model(data)
            target = logits[s_idx]
            model.zero_grad(set_to_none=True)
            if ea.grad is not None:
                ea.grad.zero_()
            target.backward(retain_graph=False)
            g = ea.grad.detach()
            grad_accum = grad_accum + g

        grad_accum = grad_accum / n_steps
        ig_edge = (orig_edge_attr - baseline_edge) * grad_accum
        ig_edge_np = ig_edge.detach().cpu().numpy().reshape(-1)

        # Aggregate IG to metabolite level, but only edges from this sample
        mask = sample_idx_arr == s_idx
        met_ids_for_s = met_idx_arr[mask]
        ig_for_s = ig_edge_np[mask]
        for m_id, ig_val in zip(met_ids_for_s, ig_for_s):
            per_sample_ig[si, m_id] += float(ig_val)

    # Restore original edge_attr
    data['sample', 'has', 'met'].edge_attr = orig_edge_attr

    return per_sample_ig, sample_indices


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
    p.add_argument('--ig-steps', type=int, default=30)
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    met_df, y, group_labels, met_cols, metadata = load_metabolomics_data(
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

    # Run 6-fold CV; compute IG on val samples of each fold so each
    # sample gets attribution from a model that did NOT train on it.
    set_all_seeds(args.seed)
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True,
                          random_state=args.seed)

    all_ig = np.full((len(y), len(met_cols)), np.nan, dtype=np.float32)
    all_probs = np.full(len(y), np.nan, dtype=np.float32)

    for fold_idx, (tr, vl) in enumerate(skf.split(X, y)):
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
                    if patience >= args.patience: break
        if best_state is not None:
            model.load_state_dict(best_state)

        # Compute probabilities for val samples
        model.eval()
        data = data.to(device)
        with torch.no_grad():
            logits = model(data)
            probs_local = torch.sigmoid(logits).cpu().numpy()
        val_local_indices = list(range(n_tr_, n_tr_ + len(vl)))
        for local_i, global_i in zip(val_local_indices, vl):
            all_probs[global_i] = float(probs_local[local_i])

        # Compute IG for val samples
        ig_arr, sample_indices_used = integrated_gradients_per_sample(
            model, data, device, n_steps=args.ig_steps,
            sample_indices=val_local_indices
        )
        for local_i, global_i in zip(val_local_indices, vl):
            local_row = sample_indices_used.index(local_i)
            all_ig[global_i] = ig_arr[local_row]
        logger.info(
            f'Fold {fold_idx + 1}/{args.n_folds}: '
            f'computed IG for {len(vl)} val samples'
        )

    # Save per-sample IG matrix (as compact CSV with sample IDs)
    df_ig = pd.DataFrame(all_ig, columns=met_cols)
    df_ig.insert(0, 'sample_id', [str(s) for s in sample_ids])
    df_ig.insert(1, 'label', y)
    df_ig.insert(2, 'predicted_prob', all_probs)
    df_ig.to_csv(outdir / 'xai_ig_per_sample.csv', index=False)

    # Per-sample top 10 drivers
    rows = []
    for i, sid in enumerate(sample_ids):
        ig_row = all_ig[i]
        if np.all(np.isnan(ig_row)):
            continue
        abs_sorted = np.argsort(np.abs(ig_row))[::-1]
        top_pos = [j for j in abs_sorted if ig_row[j] > 0][:10]
        top_neg = [j for j in abs_sorted if ig_row[j] < 0][:10]
        for rank, j in enumerate(top_pos, 1):
            rows.append({'sample_id': str(sid),
                         'label': int(y[i]),
                         'predicted_prob': float(all_probs[i]),
                         'direction': 'pushes_HIGH',
                         'rank': rank,
                         'metabolite': met_cols[j],
                         'ig_attribution': float(ig_row[j]),
                         'abundance': float(X[i, j])})
        for rank, j in enumerate(top_neg, 1):
            rows.append({'sample_id': str(sid),
                         'label': int(y[i]),
                         'predicted_prob': float(all_probs[i]),
                         'direction': 'pushes_LOW',
                         'rank': rank,
                         'metabolite': met_cols[j],
                         'ig_attribution': float(ig_row[j]),
                         'abundance': float(X[i, j])})
    pd.DataFrame(rows).to_csv(outdir / 'xai_ig_per_sample_top10.csv',
                              index=False)

    # Global importance = mean |IG| across samples
    with np.errstate(all='ignore'):
        mean_abs_ig = np.nanmean(np.abs(all_ig), axis=0)
    df_global = pd.DataFrame({
        'metabolite': met_cols,
        'mean_abs_ig': mean_abs_ig,
    }).sort_values('mean_abs_ig', ascending=False)
    df_global.to_csv(outdir / 'xai_ig_global_importance.csv', index=False)

    # Compare to raw-gradient ranking
    raw_rank = pd.read_csv(outdir / 'metabolite_rank.csv')
    raw_top30 = set(raw_rank.head(30)['metabolite'])
    ig_top30 = set(df_global.head(30)['metabolite'])
    overlap = len(raw_top30 & ig_top30)
    logger.info(
        f'\nIG global top-30 vs raw-grad top-30: overlap={overlap}/30'
    )
    logger.info(f'IG top 15:')
    for _, r in df_global.head(15).iterrows():
        logger.info(f"  {r['metabolite']:50s} {r['mean_abs_ig']:.4e}")


if __name__ == '__main__':
    main()
