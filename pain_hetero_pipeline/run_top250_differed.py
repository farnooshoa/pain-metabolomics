#!/usr/bin/env python3
"""
Retrain GNN using only the top-K most differentially-abundant metabolites
(ranked by Mann-Whitney U p-value between High and Low pain groups).

Addresses collaborator request to see a version of the model that uses
only the most-differed metabolites (feature pre-selection).

Usage:
    python run_top250_differed.py \
        --csv starting_template.csv \
        --outdir pain_hetero_pipeline/out_hetero_top250 \
        --source-dir pain_hetero_pipeline/out_hetero_endogenous_6fold \
        --top-k 250 --seed 42
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
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))

from pain_hetero_pipeline import (
    PainHeteroGNN, build_hetero_graph, train_epoch, evaluate,
    compute_metabolite_importance, compute_enzyme_scores,
)
from utils import load_metabolomics_data, preprocess_fold

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('top250')


def set_all_seeds(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--outdir', required=True)
    p.add_argument('--source-dir', required=True,
                   help='Directory with xai_wilcoxon_univariate.csv and KEGG maps')
    p.add_argument('--top-k', type=int, default=250)
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
    src = Path(args.source_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    # Load data
    met_df, y, group_labels, met_cols, metadata = load_metabolomics_data(
        args.csv, high_threshold=args.high_threshold,
        low_threshold=args.low_threshold,
        exclude_xenobiotics=args.exclude_xenobiotics,
        annotation_path=args.annotation_path,
    )
    X_full = met_df.values.astype(np.float32)
    y = np.asarray(y)
    logger.info(f'Full: X={X_full.shape}')

    # Load Wilcoxon rankings and pick top-K
    wdf = pd.read_csv(src / 'xai_wilcoxon_univariate.csv')
    wdf_sorted = wdf.sort_values('p_value').head(args.top_k)
    top_mets = set(wdf_sorted['metabolite'].tolist())
    logger.info(f'Top-{args.top_k} by Wilcoxon p-value:')
    logger.info(f'  p-value range: {wdf_sorted["p_value"].min():.2e} '
                f'to {wdf_sorted["p_value"].max():.4f}')

    # Filter X
    keep_idx = [i for i, m in enumerate(met_cols) if m in top_mets]
    X = X_full[:, keep_idx]
    met_cols_sel = [met_cols[i] for i in keep_idx]
    logger.info(f'Filtered: X={X.shape}, {len(met_cols_sel)} metabolites')

    # Load KEGG maps and filter
    with open(src / 'met2ec.json') as f: met2ec_full = json.load(f)
    with open(src / 'met2pathway.json') as f: met2pathway_full = json.load(f)
    met2ec = {m: ecs for m, ecs in met2ec_full.items() if m in top_mets}
    met2pathway = {m: p for m, p in met2pathway_full.items() if m in top_mets}
    logger.info(f'KEGG: {len(met2ec)} met2ec, {len(met2pathway)} met2pathway')

    # Save KEGG maps to outdir for later use
    with open(outdir / 'met2ec.json', 'w') as f: json.dump(met2ec, f)
    with open(outdir / 'met2pathway.json', 'w') as f: json.dump(met2pathway, f)

    # Also save which metabolites were selected + their p-values
    wdf_sorted[['metabolite', 'p_value', 'q_value', 'log2_fc',
                'median_high', 'median_low']].to_csv(
        outdir / 'selected_metabolites.csv', index=False
    )

    # 6-fold CV
    set_all_seeds(args.seed)
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True,
                          random_state=args.seed)
    fold_results = []
    all_met_importance = defaultdict(list)
    all_enz_scores = defaultdict(lambda: {'scores': [], 'metabolites': set()})

    t0 = time.time()
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
            X_all, y_all, sf_all, met_cols_sel, met2ec, met2pathway,
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

        acc, auc, f1, prec, rec, y_true, y_pred = evaluate(model, data, device)
        fold_results.append({
            'fold': fold_idx + 1, 'accuracy': acc, 'auc': auc,
            'f1': f1, 'precision': prec, 'recall': rec,
        })
        logger.info(
            f'Fold {fold_idx+1}: AUC={auc:.3f} F1={f1:.3f} Acc={acc:.3f}'
        )

        met_imp = compute_metabolite_importance(model, data, device)
        enz_sc = compute_enzyme_scores(met_imp, met2ec)
        for m, s in met_imp.items():
            all_met_importance[m].append(s)
        for ec, info in enz_sc.items():
            all_enz_scores[ec]['scores'].append(info['score'])
            for m, _ in info['metabolites']:
                all_enz_scores[ec]['metabolites'].add(m)

    # Aggregate
    cv_df = pd.DataFrame(fold_results)
    numeric_cols = ['accuracy', 'auc', 'f1', 'precision', 'recall']
    cv_df.loc[len(cv_df)] = ['mean'] + [cv_df[c].mean() for c in numeric_cols]
    cv_df.loc[len(cv_df)] = ['std'] + [cv_df[c].std() for c in numeric_cols]
    cv_df.to_csv(outdir / 'cv_metrics_overall.csv', index=False)

    # Metabolite rank
    met_rank_rows = []
    for m, scores in all_met_importance.items():
        met_rank_rows.append({
            'metabolite': m,
            'importance_score': float(np.mean(scores)),
            'importance_std': float(np.std(scores)),
            'selection_frequency': sum(1 for s in scores if s > np.mean(scores)),
        })
    met_rank_df = pd.DataFrame(met_rank_rows).sort_values(
        'importance_score', ascending=False
    )
    met_rank_df.to_csv(outdir / 'metabolite_rank.csv', index=False)

    # Enzyme rank (SUM and MEAN)
    enz_rank_rows = []
    met_imp_mean = {m: np.mean(s) for m, s in all_met_importance.items()}
    for ec, info in all_enz_scores.items():
        mets_linked = list(info['metabolites'])
        linked_imps = [met_imp_mean.get(m, 0) for m in mets_linked]
        enz_rank_rows.append({
            'enzyme_id': ec,
            'enzyme_score': float(np.mean(info['scores'])),
            'score_std': float(np.std(info['scores'])),
            'degree': len(mets_linked),
            'score_MEAN_importance': float(np.mean(linked_imps)) if linked_imps else 0,
            'score_MAX_importance': float(np.max(linked_imps)) if linked_imps else 0,
            'supporting_metabolites': '; '.join(mets_linked[:5]),
            'selection_frequency': len(info['scores']),
        })
    enz_rank_df = pd.DataFrame(enz_rank_rows).sort_values(
        'enzyme_score', ascending=False
    )
    enz_rank_df.to_csv(outdir / 'enzyme_rank.csv', index=False)

    logger.info('=' * 60)
    logger.info('TOP-250 DIFFERED MODEL RESULTS')
    logger.info('=' * 60)
    logger.info(
        f'AUC: {cv_df[cv_df["fold"]=="mean"]["auc"].values[0]:.3f} '
        f'± {cv_df[cv_df["fold"]=="std"]["auc"].values[0]:.3f}'
    )
    logger.info(
        f'F1:  {cv_df[cv_df["fold"]=="mean"]["f1"].values[0]:.3f}'
    )
    logger.info(f'Top 15 enzymes (SUM):')
    for _, r in enz_rank_df.head(15).iterrows():
        logger.info(
            f"  {r['enzyme_id']:<15s} score={r['enzyme_score']:.2e} "
            f"degree={r['degree']}"
        )
    logger.info(f'Top 15 metabolites:')
    for _, r in met_rank_df.head(15).iterrows():
        logger.info(f"  {r['metabolite'][:50]:50s} {r['importance_score']:.2e}")
    logger.info(f'\nOutputs: {outdir}')
    logger.info(f'Elapsed: {(time.time() - t0)/60:.1f} min')


if __name__ == '__main__':
    main()
