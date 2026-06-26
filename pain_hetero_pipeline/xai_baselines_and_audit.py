#!/usr/bin/env python3
"""
XAI Tier 1, Part A: Baseline model comparison + univariate dataset audit.

Outputs into --outdir:
    xai_baseline_models.csv     : per-model per-fold AUC, F1, etc.
    xai_baseline_top_features.csv: top-30 features from each baseline
    xai_wilcoxon_univariate.csv : per-metabolite Wilcoxon High vs Low + BH-FDR
    xai_top30_enzymes_univariate.csv: union of supporting metabolites for Top-30 enzymes,
                                      with univariate stats
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
from utils import load_metabolomics_data

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('xai_bl')


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


def run_baseline(clf_name, make_clf, X, y, n_folds, seed):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    results = []
    feat_imp = np.zeros(X.shape[1])
    for fold_idx, (tr, vl) in enumerate(skf.split(X, y)):
        X_tr, X_vl = X[tr], X[vl]
        y_tr, y_vl = y[tr], y[vl]
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_vl_s = sc.transform(X_vl)
        clf = make_clf()
        clf.fit(X_tr_s, y_tr)
        if hasattr(clf, 'predict_proba'):
            pred = clf.predict_proba(X_vl_s)[:, 1]
        else:
            pred = clf.decision_function(X_vl_s)
        y_bin = (pred > 0.5).astype(int) if hasattr(clf, 'predict_proba') else clf.predict(X_vl_s)
        try:
            auc = roc_auc_score(y_vl, pred)
        except Exception:
            auc = np.nan
        acc = accuracy_score(y_vl, y_bin)
        f1 = f1_score(y_vl, y_bin, zero_division=0)
        results.append({'model': clf_name, 'fold': fold_idx + 1,
                        'auc': auc, 'accuracy': acc, 'f1': f1})
        # Feature importance
        if hasattr(clf, 'coef_'):
            feat_imp += np.abs(clf.coef_[0])
        elif hasattr(clf, 'feature_importances_'):
            feat_imp += clf.feature_importances_
    feat_imp /= n_folds
    return results, feat_imp


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--outdir', required=True)
    p.add_argument('--n-folds', type=int, default=6)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--high-threshold', type=int, default=4)
    p.add_argument('--low-threshold', type=int, default=3)
    p.add_argument('--exclude-xenobiotics', action='store_true')
    p.add_argument('--annotation-path', type=str,
                   default='metabolites names and pathways.xlsx')
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    met_df, y, group_labels, met_cols, metadata = load_metabolomics_data(
        args.csv, high_threshold=args.high_threshold,
        low_threshold=args.low_threshold,
        exclude_xenobiotics=args.exclude_xenobiotics,
        annotation_path=args.annotation_path,
    )
    X = met_df.values.astype(np.float32)
    y = np.asarray(y)
    logger.info(f'X={X.shape}, y={y.shape}')

    # -------- Baseline models --------
    logger.info('=' * 60)
    logger.info('Running baseline models (6-fold CV)...')
    logger.info('=' * 60)

    configs = [
        ('LogReg_L1',
         lambda: LogisticRegression(penalty='l1', solver='liblinear',
                                    C=0.1, max_iter=1000,
                                    random_state=args.seed)),
        ('LogReg_L2',
         lambda: LogisticRegression(penalty='l2', C=0.1, max_iter=1000,
                                    random_state=args.seed)),
        ('RandomForest',
         lambda: RandomForestClassifier(n_estimators=300, max_depth=5,
                                        random_state=args.seed, n_jobs=-1)),
    ]
    try:
        from xgboost import XGBClassifier
        configs.append(
            ('XGBoost',
             lambda: XGBClassifier(n_estimators=200, max_depth=4,
                                   learning_rate=0.05,
                                   use_label_encoder=False,
                                   eval_metric='logloss',
                                   random_state=args.seed, verbosity=0,
                                   n_jobs=-1))
        )
    except Exception as e:
        logger.warning(f'xgboost unavailable, skipping ({e})')

    all_res = []
    all_imp = {}
    for name, mk in configs:
        try:
            res, imp = run_baseline(name, mk, X, y, args.n_folds, args.seed)
            all_res.extend(res)
            all_imp[name] = imp
            aucs = [r['auc'] for r in res if not np.isnan(r['auc'])]
            f1s = [r['f1'] for r in res]
            accs = [r['accuracy'] for r in res]
            logger.info(
                f'{name:15s} AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f}, '
                f'Acc={np.mean(accs):.3f}, F1={np.mean(f1s):.3f}'
            )
        except Exception as e:
            logger.error(f'{name} failed: {e}')

    df_bl = pd.DataFrame(all_res)
    summary = df_bl.groupby('model').agg(
        auc_mean=('auc', 'mean'), auc_std=('auc', 'std'),
        acc_mean=('accuracy', 'mean'), f1_mean=('f1', 'mean'),
    ).reset_index()
    df_bl.to_csv(outdir / 'xai_baseline_models.csv', index=False)
    summary.to_csv(outdir / 'xai_baseline_summary.csv', index=False)

    # Top 30 features per baseline
    rows = []
    for name, imp in all_imp.items():
        order = np.argsort(imp)[::-1]
        for rank, i in enumerate(order[:30], 1):
            rows.append({'model': name, 'rank': rank,
                         'metabolite': met_cols[i],
                         'importance': float(imp[i])})
    pd.DataFrame(rows).to_csv(outdir / 'xai_baseline_top_features.csv',
                              index=False)
    logger.info(f'Wrote baseline CSVs to {outdir}')

    # Overlap with GNN top 30
    gnn_met = pd.read_csv(outdir / 'metabolite_rank.csv')
    gnn_top30 = set(gnn_met.head(30)['metabolite'])
    gnn_top100 = set(gnn_met.head(100)['metabolite'])
    logger.info('\nFeature overlap with GNN top-30:')
    for name, imp in all_imp.items():
        order = np.argsort(imp)[::-1]
        top30 = set(met_cols[i] for i in order[:30])
        top100 = set(met_cols[i] for i in order[:100])
        o30 = len(top30 & gnn_top30)
        o100 = len(top100 & gnn_top100)
        logger.info(f'  {name:15s} Top30∩: {o30}/30, Top100∩: {o100}/100')

    # -------- Univariate Wilcoxon --------
    logger.info('=' * 60)
    logger.info('Running univariate Wilcoxon (Mann-Whitney U)...')
    logger.info('=' * 60)
    pvals = []
    stats = []
    medians_hi = []
    medians_lo = []
    mask_hi = y == 1
    mask_lo = y == 0
    for i, m in enumerate(met_cols):
        xi = X[:, i]
        try:
            s, p = mannwhitneyu(xi[mask_hi], xi[mask_lo],
                                alternative='two-sided')
        except Exception:
            s, p = np.nan, np.nan
        stats.append(s); pvals.append(p)
        medians_hi.append(float(np.median(xi[mask_hi])))
        medians_lo.append(float(np.median(xi[mask_lo])))
    pvals = np.array(pvals)
    qvals = np.full_like(pvals, np.nan)
    valid = ~np.isnan(pvals)
    if valid.any():
        qvals[valid] = bh_fdr(pvals[valid])
    wdf = pd.DataFrame({
        'metabolite': met_cols,
        'median_high': medians_hi,
        'median_low': medians_lo,
        'log2_fc': [np.log2((h + 1e-10) / (l + 1e-10)) for h, l in zip(medians_hi, medians_lo)],
        'wilcoxon_stat': stats,
        'p_value': pvals,
        'q_value': qvals,
        'significant_fdr': qvals < 0.05,
    }).sort_values('p_value')
    wdf.to_csv(outdir / 'xai_wilcoxon_univariate.csv', index=False)

    n_sig = int(wdf['significant_fdr'].sum())
    n_p05 = int((wdf['p_value'] < 0.05).sum())
    logger.info(
        f'Wilcoxon: {n_p05}/{len(wdf)} mets with p<0.05 (uncorrected), '
        f'{n_sig}/{len(wdf)} with q<0.05 (FDR)'
    )

    # Top 30 enzymes -> supporting metabolites univariate check
    enz_df = pd.read_csv(outdir / 'enzyme_rank.csv')
    top30_enz = enz_df.head(30)
    supporting_sets = []
    for _, r in top30_enz.iterrows():
        mets = [m.strip() for m in str(r.get('supporting_metabolites', '')).split(';') if m.strip()]
        supporting_sets.append((r['enzyme_id'], mets))
    rows_enz = []
    wd = wdf.set_index('metabolite')
    for ec, mets in supporting_sets:
        for m in mets:
            if m in wd.index:
                r = wd.loc[m]
                rows_enz.append({
                    'enzyme_id': ec, 'metabolite': m,
                    'p_value': r['p_value'], 'q_value': r['q_value'],
                    'log2_fc': r['log2_fc'],
                    'significant_fdr': bool(r['significant_fdr']),
                })
    pd.DataFrame(rows_enz).to_csv(
        outdir / 'xai_top30_enzymes_univariate.csv', index=False
    )

    n_sig_in_top = int(pd.DataFrame(rows_enz)['significant_fdr'].sum())
    n_total_in_top = len(rows_enz)
    logger.info(
        f'Top-30 enzymes supporting metabolites: '
        f'{n_sig_in_top}/{n_total_in_top} are univariately FDR-significant'
    )

    logger.info(f'All outputs in {outdir}')


if __name__ == '__main__':
    main()
