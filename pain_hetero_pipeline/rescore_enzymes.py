#!/usr/bin/env python3
"""
Re-score enzymes using alternative aggregation formulas to address the
degree bias identified by the wet-lab collaborator.

Current: enzyme_score(ec) = SUM of metabolite_importance of linked mets
        -> rewards high-degree enzymes mechanically (r=0.994 with degree)

Alternatives computed:
    MEAN              : mean importance of linked mets
    TOP3_MEAN         : mean importance of top-3 linked mets
    TOP5_MEAN         : mean importance of top-5 linked mets
    MAX               : max importance of linked mets (single best)
    SUM_NORM_SQRT     : SUM / sqrt(degree)
    RANK_WEIGHTED     : sum(1 / rank of each linked met)
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--outdir', required=True)
    args = p.parse_args()
    outdir = Path(args.outdir)

    # Load
    met = pd.read_csv(outdir / 'metabolite_rank.csv')
    with open(outdir / 'met2ec.json') as f:
        met2ec = json.load(f)

    met_imp = dict(zip(met['metabolite'], met['importance_score']))
    met_rank = {m: i + 1 for i, m in enumerate(met['metabolite'])}

    # Build ec -> list of (met, importance, rank)
    ec2linked = defaultdict(list)
    for m, ecs in met2ec.items():
        if m not in met_imp:
            continue
        for ec in ecs:
            ec2linked[ec].append((m, met_imp[m], met_rank[m]))

    # Compute all scoring variants
    rows = []
    for ec, linked in ec2linked.items():
        imps = sorted([l[1] for l in linked], reverse=True)
        ranks = sorted([l[2] for l in linked])
        n = len(imps)
        if n == 0:
            continue
        rows.append({
            'enzyme_id': ec,
            'degree': n,
            'best_linked_rank': ranks[0],
            'median_linked_rank': float(np.median(ranks)),
            'score_SUM':           float(np.sum(imps)),
            'score_MEAN':          float(np.mean(imps)),
            'score_TOP3_MEAN':     float(np.mean(imps[:3])),
            'score_TOP5_MEAN':     float(np.mean(imps[:5])),
            'score_MAX':           float(imps[0]),
            'score_SUM_NORM_SQRT': float(np.sum(imps) / np.sqrt(n)),
            'score_RANK_WEIGHTED': float(np.sum([1.0 / r for r in ranks])),
        })

    df = pd.DataFrame(rows)

    # Correlation of each score with degree
    print('=' * 70)
    print('Correlation of each score with DEGREE (goal: low correlation)')
    print('=' * 70)
    score_cols = [c for c in df.columns if c.startswith('score_')]
    for c in score_cols:
        r, _ = spearmanr(df['degree'], df[c])
        print(f'  {c:<25s}  Spearman r with degree = {r:+.3f}')

    # Save per-method top 30
    print('\n' + '=' * 70)
    print('TOP 15 ENZYMES by each scoring formula')
    print('=' * 70)
    all_top30 = {}
    for c in score_cols:
        method = c.replace('score_', '')
        ranked = df.sort_values(c, ascending=False).reset_index(drop=True)
        all_top30[method] = ranked.head(30)['enzyme_id'].tolist()
        print(f'\n--- {method} ---')
        for i, row in ranked.head(15).iterrows():
            print(f'  {i+1:3d}  {row["enzyme_id"]:<15s} '
                  f'degree={row["degree"]:3d}  '
                  f'best_rank={int(row["best_linked_rank"]):4d}  '
                  f'score={row[c]:.3e}')

    # Overlap matrix between methods
    print('\n' + '=' * 70)
    print('TOP-30 OVERLAP MATRIX')
    print('=' * 70)
    methods = list(all_top30.keys())
    print(f'{"":<20s} ' + ' '.join(f'{m:<15s}' for m in methods))
    for m1 in methods:
        overlaps = [len(set(all_top30[m1]) & set(all_top30[m2])) for m2 in methods]
        print(f'{m1:<20s} ' + ' '.join(f'{v:<15d}' for v in overlaps))

    # Write merged comparison CSV
    df_sorted = df.sort_values('score_SUM', ascending=False).reset_index(drop=True)
    df_sorted.insert(0, 'rank_SUM', df_sorted.index + 1)
    for c in score_cols:
        method = c.replace('score_', '')
        rk_col = f'rank_{method}'
        df_sorted[rk_col] = df_sorted[c].rank(ascending=False, method='min').astype(int)
    df_sorted.to_csv(outdir / 'enzyme_rank_alt_scoring.csv', index=False)

    # Pain-relevant known enzymes (from v1.0 external validation)
    known_pain = {
        'ec:1.17.3.2': 'XDH/XO',
        'ec:1.17.1.4': 'XDH (NAD+ form)',
        'ec:3.2.1.22': 'GLA (Fabry)',
        'ec:3.1.6.2': 'STS',
        'ec:2.8.2.14': 'SULT2A1',
        'ec:3.1.3.5': 'ACP3',
        'ec:1.2.1.3': 'ALDH1A1',
        'ec:1.2.3.1': 'AOX1',
    }
    print('\n' + '=' * 70)
    print('RECOVERY of v1.0 known pain-relevant enzymes in each method top-30')
    print('=' * 70)
    for method in methods:
        top30_set = set(all_top30[method])
        hits = [ec for ec in known_pain if ec in top30_set]
        print(f'  {method:<20s} {len(hits)}/8 hits: {hits}')

    print(f'\nSaved: {outdir / "enzyme_rank_alt_scoring.csv"}')


if __name__ == '__main__':
    main()
