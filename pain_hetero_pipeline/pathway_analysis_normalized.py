#!/usr/bin/env python3
"""
Pathway-Level Analysis with Normalized Scoring

Computes:
1. Mean enzyme score (size-normalized)
2. Top-10 enzyme sum score
3. Excludes generic superpathways
4. Outputs top candidate pathways with detailed enzyme/protein/metabolite info
"""

import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict
import requests
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

OUT_DIR = "out_hetero"
KEGG_CACHE_DIR = os.path.join(OUT_DIR, "kegg_cache")

# Generic superpathways to exclude (too broad to be interpretable)
EXCLUDE_PATHWAYS = {
    'path:map01100',  # Metabolic pathways
    'path:map01110',  # Biosynthesis of secondary metabolites
    'path:map01120',  # Microbial metabolism in diverse environments
    'path:map01200',  # Carbon metabolism
    'path:map01230',  # Biosynthesis of amino acids
    'path:map01232',  # Nucleotide metabolism (broad)
    'path:map01250',  # Biosynthesis of nucleotide sugars
}

# Drug patterns for filtering
DRUG_PATTERNS = [
    'metformin', 'metoprolol', 'diazepam', 'glipizide', 'citalopram',
    'atorvastatin', 'zolpidem', 'famotidine', 'dextromethorphan',
    'chlorothiazide', 'THC', 'caffeine', 'theobromine', 'paraxanthine',
    'resveratrol', 'glucuronide', 'acetaminophen', 'ibuprofen',
    'aspirin', 'warfarin', 'omeprazole', 'losartan', 'lisinopril',
    'simvastatin', 'pravastatin', 'gabapentin', 'pregabalin', 'tramadol',
    'oxycodone', 'hydrocodone', 'morphine', 'codeine', 'fentanyl',
    'nicotine', 'cotinine'
]

# KEGG pathway names
KEGG_PATHWAY_NAMES = {
    'map00010': 'Glycolysis / Gluconeogenesis',
    'map00020': 'Citrate cycle (TCA cycle)',
    'map00030': 'Pentose phosphate pathway',
    'map00040': 'Pentose and glucuronate interconversions',
    'map00053': 'Ascorbate and aldarate metabolism',
    'map00071': 'Fatty acid degradation',
    'map00100': 'Steroid biosynthesis',
    'map00120': 'Primary bile acid biosynthesis',
    'map00140': 'Steroid hormone biosynthesis',
    'map00190': 'Oxidative phosphorylation',
    'map00220': 'Arginine biosynthesis',
    'map00230': 'Purine metabolism',
    'map00232': 'Caffeine metabolism',
    'map00240': 'Pyrimidine metabolism',
    'map00250': 'Alanine, aspartate and glutamate metabolism',
    'map00260': 'Glycine, serine and threonine metabolism',
    'map00270': 'Cysteine and methionine metabolism',
    'map00280': 'Valine, leucine and isoleucine degradation',
    'map00310': 'Lysine degradation',
    'map00330': 'Arginine and proline metabolism',
    'map00340': 'Histidine metabolism',
    'map00350': 'Tyrosine metabolism',
    'map00360': 'Phenylalanine metabolism',
    'map00380': 'Tryptophan metabolism',
    'map00400': 'Phenylalanine, tyrosine and tryptophan biosynthesis',
    'map00410': 'beta-Alanine metabolism',
    'map00430': 'Taurine and hypotaurine metabolism',
    'map00470': 'D-Amino acid metabolism',
    'map00480': 'Glutathione metabolism',
    'map00500': 'Starch and sucrose metabolism',
    'map00520': 'Amino sugar and nucleotide sugar metabolism',
    'map00561': 'Glycerolipid metabolism',
    'map00562': 'Inositol phosphate metabolism',
    'map00564': 'Glycerophospholipid metabolism',
    'map00590': 'Arachidonic acid metabolism',
    'map00600': 'Sphingolipid metabolism',
    'map00620': 'Pyruvate metabolism',
    'map00630': 'Glyoxylate and dicarboxylate metabolism',
    'map00640': 'Propanoate metabolism',
    'map00650': 'Butanoate metabolism',
    'map00670': 'One carbon pool by folate',
    'map00730': 'Thiamine metabolism',
    'map00740': 'Riboflavin metabolism',
    'map00750': 'Vitamin B6 metabolism',
    'map00760': 'Nicotinate and nicotinamide metabolism',
    'map00770': 'Pantothenate and CoA biosynthesis',
    'map00860': 'Porphyrin metabolism',
    'map00900': 'Terpenoid backbone biosynthesis',
    'map00908': 'Zeatin biosynthesis',
    'map00910': 'Nitrogen metabolism',
    'map00920': 'Sulfur metabolism',
    'map00980': 'Metabolism of xenobiotics by cytochrome P450',
    'map00982': 'Drug metabolism - cytochrome P450',
    'map00983': 'Drug metabolism - other enzymes',
    'map00999': 'Biosynthesis of various plant secondary metabolites',
    'map01040': 'Biosynthesis of unsaturated fatty acids',
    'map02010': 'ABC transporters',
    'map04024': 'cAMP signaling pathway',
    'map04270': 'Vascular smooth muscle contraction',
    'map04713': 'Circadian entrainment',
    'map04727': 'GABAergic synapse',
    'map04913': 'Ovarian steroidogenesis',
    'map04923': 'Regulation of lipolysis in adipocytes',
    'map04976': 'Bile secretion',
    'map04979': 'Cholesterol metabolism',
}


def get_pathway_name(pathway_id: str) -> str:
    """Get human-readable pathway name."""
    map_id = pathway_id.replace('path:', '')
    if map_id in KEGG_PATHWAY_NAMES:
        return KEGG_PATHWAY_NAMES[map_id]
    return map_id


def is_drug_related(met_name: str) -> bool:
    """Check if metabolite is drug/xenobiotic related."""
    name_lower = met_name.lower()
    for pattern in DRUG_PATTERNS:
        if pattern.lower() in name_lower:
            return True
    return False


def fetch_enzyme_genes(ec_number: str) -> list:
    """Fetch human genes associated with an enzyme from KEGG."""
    ec_id = ec_number.replace('ec:', '')
    cache_file = Path(KEGG_CACHE_DIR) / f"enzyme_genes_{ec_id.replace('.', '_')}.json"

    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except:
            pass

    try:
        url = f"https://rest.kegg.jp/link/hsa/ec:{ec_id}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            genes = []
            for line in response.text.strip().split('\n'):
                if '\t' in line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        genes.append(parts[1])
            Path(KEGG_CACHE_DIR).mkdir(exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump(genes, f)
            time.sleep(0.15)
            return genes
    except:
        pass
    return []


def fetch_gene_symbol(gene_id: str) -> str:
    """Fetch gene symbol from KEGG gene ID."""
    cache_file = Path(KEGG_CACHE_DIR) / f"gene_{gene_id.replace(':', '_')}.json"

    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
                return data.get('symbol', gene_id)
        except:
            pass

    try:
        url = f"https://rest.kegg.jp/get/{gene_id}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            symbol = gene_id
            for line in response.text.split('\n'):
                if line.startswith('SYMBOL'):
                    symbol = line.split()[1] if len(line.split()) > 1 else gene_id
                    break
            Path(KEGG_CACHE_DIR).mkdir(exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump({'symbol': symbol}, f)
            time.sleep(0.15)
            return symbol
    except:
        pass
    return gene_id


# =============================================================================
# Main Analysis
# =============================================================================

def main():
    print("=" * 70)
    print("PATHWAY ANALYSIS WITH NORMALIZED SCORING")
    print("=" * 70)

    # Load data
    enzyme_df = pd.read_csv(os.path.join(OUT_DIR, 'enzyme_rank.csv'))
    met_df = pd.read_csv(os.path.join(OUT_DIR, 'metabolite_rank.csv'))

    print(f"Loaded {len(enzyme_df)} enzymes, {len(met_df)} metabolites")

    # Identify drug-related metabolites
    drug_mets = set()
    for met in met_df['metabolite']:
        if is_drug_related(str(met)):
            drug_mets.add(str(met).lower())

    print(f"Drug-related metabolites: {len(drug_mets)}")

    # Build pathway data with enzyme-level detail
    pathway_data = defaultdict(lambda: {
        'enzyme_ids': [],
        'enzyme_scores': [],
        'metabolites': set(),
        'endo_metabolites': set()
    })

    for _, row in enzyme_df.iterrows():
        ec_id = row['enzyme_id']
        score = row['enzyme_score']

        # Parse pathways
        pathways_str = str(row.get('pathways', ''))
        if not pathways_str or pathways_str == 'nan':
            continue

        pathways = [p.strip() for p in pathways_str.split(';') if p.strip()]

        # Parse metabolites
        mets_str = str(row.get('supporting_metabolites', ''))
        mets = [m.strip() for m in mets_str.split(';') if m.strip()]
        endo_mets = [m for m in mets if m.lower() not in drug_mets]

        for pathway in pathways:
            # Skip excluded superpathways
            if pathway in EXCLUDE_PATHWAYS:
                continue

            pathway_data[pathway]['enzyme_ids'].append(ec_id)
            pathway_data[pathway]['enzyme_scores'].append(score)
            pathway_data[pathway]['metabolites'].update(mets)
            pathway_data[pathway]['endo_metabolites'].update(endo_mets)

    print(f"Found {len(pathway_data)} pathways (after excluding superpathways)")

    # Compute normalized scores
    pathway_rows = []

    for pathway_id, data in pathway_data.items():
        n_enzymes = len(data['enzyme_ids'])
        if n_enzymes == 0:
            continue

        scores = np.array(data['enzyme_scores'])

        # Mean score (size-normalized)
        mean_score = np.mean(scores)

        # Top-10 sum score
        top10_scores = np.sort(scores)[::-1][:10]
        top10_sum = np.sum(top10_scores)

        # Get top enzymes for this pathway
        sorted_idx = np.argsort(scores)[::-1]
        top_enzymes = [data['enzyme_ids'][i] for i in sorted_idx[:10]]
        top_enzyme_scores = [scores[i] for i in sorted_idx[:10]]

        pathway_rows.append({
            'pathway_id': pathway_id,
            'pathway_name': get_pathway_name(pathway_id),
            'mean_enzyme_score': mean_score,
            'top10_sum_score': top10_sum,
            'total_sum_score': np.sum(scores),
            'n_enzymes': n_enzymes,
            'n_metabolites': len(data['metabolites']),
            'n_endo_metabolites': len(data['endo_metabolites']),
            'top_enzymes': top_enzymes,
            'top_enzyme_scores': top_enzyme_scores,
            'supporting_metabolites': list(data['endo_metabolites'])
        })

    df = pd.DataFrame(pathway_rows)

    # ==========================================================================
    # Output 1: pathway_rank_normalized_mean.csv (ranked by mean score)
    # ==========================================================================
    df_mean = df.sort_values('mean_enzyme_score', ascending=False).reset_index(drop=True)
    df_mean_out = df_mean[['pathway_id', 'pathway_name', 'mean_enzyme_score',
                           'top10_sum_score', 'n_enzymes', 'n_endo_metabolites']].copy()
    df_mean_out.to_csv(os.path.join(OUT_DIR, 'pathway_rank_normalized_mean.csv'), index=False)
    print(f"\nSaved pathway_rank_normalized_mean.csv")

    print("\nTop 10 Pathways by MEAN enzyme score:")
    for i, row in df_mean_out.head(10).iterrows():
        print(f"  {i+1}. {row['pathway_name'][:50]:50s} mean={row['mean_enzyme_score']:.2e} n={row['n_enzymes']}")

    # ==========================================================================
    # Output 2: pathway_rank_top10.csv (ranked by top-10 sum)
    # ==========================================================================
    df_top10 = df.sort_values('top10_sum_score', ascending=False).reset_index(drop=True)
    df_top10_out = df_top10[['pathway_id', 'pathway_name', 'top10_sum_score',
                             'mean_enzyme_score', 'n_enzymes', 'n_endo_metabolites']].copy()
    df_top10_out.to_csv(os.path.join(OUT_DIR, 'pathway_rank_top10.csv'), index=False)
    print(f"\nSaved pathway_rank_top10.csv")

    print("\nTop 10 Pathways by TOP-10 enzyme sum:")
    for i, row in df_top10_out.head(10).iterrows():
        print(f"  {i+1}. {row['pathway_name'][:50]:50s} top10={row['top10_sum_score']:.2e} n={row['n_enzymes']}")

    # ==========================================================================
    # Output 3: top_candidate_paths.csv (detailed top 5 pathways)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("GENERATING TOP CANDIDATE PATHWAYS WITH DETAILS")
    print("=" * 70)

    # Use mean score for final ranking (more interpretable)
    top5_pathways = df_mean.head(5)

    candidate_rows = []

    for rank, (_, pathway) in enumerate(top5_pathways.iterrows(), 1):
        print(f"\nProcessing #{rank}: {pathway['pathway_name']}")

        # Get top 10 enzymes
        top_enzymes = pathway['top_enzymes'][:10]
        top_scores = pathway['top_enzyme_scores'][:10]

        # Fetch proteins/genes for each enzyme
        all_proteins = []
        enzyme_protein_map = {}

        for ec in top_enzymes:
            genes = fetch_enzyme_genes(ec)
            symbols = []
            for g in genes[:5]:  # Limit to 5 genes per enzyme
                sym = fetch_gene_symbol(g)
                symbols.append(sym)
            enzyme_protein_map[ec] = symbols
            all_proteins.extend(symbols)

        # Get top metabolites
        top_mets = sorted(pathway['supporting_metabolites'],
                         key=lambda m: met_df[met_df['metabolite'] == m]['importance_score'].values[0]
                         if m in met_df['metabolite'].values else 0,
                         reverse=True)[:10]

        candidate_rows.append({
            'rank': rank,
            'pathway_id': pathway['pathway_id'],
            'pathway_name': pathway['pathway_name'],
            'mean_enzyme_score': pathway['mean_enzyme_score'],
            'top10_sum_score': pathway['top10_sum_score'],
            'n_enzymes': pathway['n_enzymes'],
            'n_metabolites': pathway['n_endo_metabolites'],
            'top10_enzymes': '; '.join(top_enzymes),
            'top10_enzyme_scores': '; '.join([f"{s:.2e}" for s in top_scores]),
            'top10_proteins': '; '.join(list(set(all_proteins))[:10]),
            'enzyme_protein_mapping': str(enzyme_protein_map),
            'top10_metabolites': '; '.join(top_mets)
        })

        # Print details
        print(f"  Enzymes: {', '.join(top_enzymes[:5])}...")
        print(f"  Proteins: {', '.join(list(set(all_proteins))[:5])}...")
        print(f"  Metabolites: {', '.join(top_mets[:5])}...")

    candidate_df = pd.DataFrame(candidate_rows)
    candidate_df.to_csv(os.path.join(OUT_DIR, 'top_candidate_paths.csv'), index=False)
    print(f"\nSaved top_candidate_paths.csv")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("FINAL TOP 5 CANDIDATE PATHWAYS")
    print("=" * 70)

    for _, row in candidate_df.iterrows():
        print(f"\n#{row['rank']}. {row['pathway_name']}")
        print(f"    ID: {row['pathway_id']}")
        print(f"    Mean Score: {row['mean_enzyme_score']:.2e}")
        print(f"    Top-10 Sum: {row['top10_sum_score']:.2e}")
        print(f"    Enzymes: {row['n_enzymes']}, Metabolites: {row['n_metabolites']}")
        enzymes = row['top10_enzymes'].split('; ')[:5]
        print(f"    Top Enzymes: {', '.join(enzymes)}")
        proteins = row['top10_proteins'].split('; ')[:5]
        print(f"    Top Proteins: {', '.join(proteins)}")
        mets = row['top10_metabolites'].split('; ')[:5]
        print(f"    Top Metabolites: {', '.join(mets)}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
