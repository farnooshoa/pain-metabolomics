#!/usr/bin/env python3
"""
Pathway-Level Aggregation and Visualization

Post-hoc analysis based on pre-computed GNN enzyme rankings.
Aggregates enzyme scores to pathway level and generates visualizations.
"""

import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx
import requests
import time
import re
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

OUT_DIR = "out_hetero"
VIZ_DIR = os.path.join(OUT_DIR, "gnn_viz")
KEGG_CACHE_DIR = os.path.join(OUT_DIR, "kegg_cache")

# Common drug/xenobiotic patterns
DRUG_PATTERNS = [
    r'metformin', r'metoprolol', r'diazepam', r'glipizide', r'citalopram',
    r'atorvastatin', r'zolpidem', r'famotidine', r'dextromethorphan',
    r'chlorothiazide', r'THC', r'caffeine', r'theobromine', r'paraxanthine',
    r'resveratrol', r'glucuronide', r'sulfate$', r'acetaminophen', r'ibuprofen',
    r'aspirin', r'warfarin', r'omeprazole', r'losartan', r'lisinopril',
    r'simvastatin', r'pravastatin', r'gabapentin', r'pregabalin', r'tramadol',
    r'oxycodone', r'hydrocodone', r'morphine', r'codeine', r'fentanyl',
    r'benzodiazepine', r'barbiturate', r'amphetamine', r'cocaine',
    r'nicotine', r'cotinine', r'ethanol', r'acetaldehyde'
]

# KEGG pathway names (partial list, will be fetched from API)
KEGG_PATHWAY_NAMES = {
    'map00010': 'Glycolysis / Gluconeogenesis',
    'map00020': 'Citrate cycle (TCA cycle)',
    'map00030': 'Pentose phosphate pathway',
    'map00040': 'Pentose and glucuronate interconversions',
    'map00051': 'Fructose and mannose metabolism',
    'map00052': 'Galactose metabolism',
    'map00053': 'Ascorbate and aldarate metabolism',
    'map00061': 'Fatty acid biosynthesis',
    'map00062': 'Fatty acid elongation',
    'map00071': 'Fatty acid degradation',
    'map00100': 'Steroid biosynthesis',
    'map00120': 'Primary bile acid biosynthesis',
    'map00130': 'Ubiquinone and other terpenoid-quinone biosynthesis',
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
    'map00290': 'Valine, leucine and isoleucine biosynthesis',
    'map00300': 'Lysine biosynthesis',
    'map00310': 'Lysine degradation',
    'map00330': 'Arginine and proline metabolism',
    'map00340': 'Histidine metabolism',
    'map00350': 'Tyrosine metabolism',
    'map00360': 'Phenylalanine metabolism',
    'map00380': 'Tryptophan metabolism',
    'map00400': 'Phenylalanine, tyrosine and tryptophan biosynthesis',
    'map00410': 'beta-Alanine metabolism',
    'map00430': 'Taurine and hypotaurine metabolism',
    'map00440': 'Phosphonate and phosphinate metabolism',
    'map00450': 'Selenocompound metabolism',
    'map00460': 'Cyanoamino acid metabolism',
    'map00470': 'D-Amino acid metabolism',
    'map00480': 'Glutathione metabolism',
    'map00500': 'Starch and sucrose metabolism',
    'map00510': 'N-Glycan biosynthesis',
    'map00520': 'Amino sugar and nucleotide sugar metabolism',
    'map00524': 'Neomycin, kanamycin and gentamicin biosynthesis',
    'map00531': 'Glycosaminoglycan degradation',
    'map00561': 'Glycerolipid metabolism',
    'map00562': 'Inositol phosphate metabolism',
    'map00563': 'Glycosylphosphatidylinositol (GPI)-anchor biosynthesis',
    'map00564': 'Glycerophospholipid metabolism',
    'map00565': 'Ether lipid metabolism',
    'map00590': 'Arachidonic acid metabolism',
    'map00591': 'Linoleic acid metabolism',
    'map00592': 'alpha-Linolenic acid metabolism',
    'map00600': 'Sphingolipid metabolism',
    'map00620': 'Pyruvate metabolism',
    'map00630': 'Glyoxylate and dicarboxylate metabolism',
    'map00640': 'Propanoate metabolism',
    'map00650': 'Butanoate metabolism',
    'map00660': 'C5-Branched dibasic acid metabolism',
    'map00670': 'One carbon pool by folate',
    'map00680': 'Methane metabolism',
    'map00730': 'Thiamine metabolism',
    'map00740': 'Riboflavin metabolism',
    'map00750': 'Vitamin B6 metabolism',
    'map00760': 'Nicotinate and nicotinamide metabolism',
    'map00770': 'Pantothenate and CoA biosynthesis',
    'map00780': 'Biotin metabolism',
    'map00785': 'Lipoic acid metabolism',
    'map00790': 'Folate biosynthesis',
    'map00860': 'Porphyrin metabolism',
    'map00900': 'Terpenoid backbone biosynthesis',
    'map00901': 'Indole alkaloid biosynthesis',
    'map00902': 'Monoterpenoid biosynthesis',
    'map00903': 'Limonene degradation',
    'map00904': 'Diterpenoid biosynthesis',
    'map00905': 'Brassinosteroid biosynthesis',
    'map00906': 'Carotenoid biosynthesis',
    'map00908': 'Zeatin biosynthesis',
    'map00910': 'Nitrogen metabolism',
    'map00920': 'Sulfur metabolism',
    'map00970': 'Aminoacyl-tRNA biosynthesis',
    'map00980': 'Metabolism of xenobiotics by cytochrome P450',
    'map00982': 'Drug metabolism - cytochrome P450',
    'map00983': 'Drug metabolism - other enzymes',
    'map00999': 'Biosynthesis of various plant secondary metabolites',
    'map01040': 'Biosynthesis of unsaturated fatty acids',
    'map01100': 'Metabolic pathways',
    'map01110': 'Biosynthesis of secondary metabolites',
    'map01120': 'Microbial metabolism in diverse environments',
    'map01200': 'Carbon metabolism',
    'map01210': '2-Oxocarboxylic acid metabolism',
    'map01212': 'Fatty acid metabolism',
    'map01230': 'Biosynthesis of amino acids',
    'map01232': 'Nucleotide metabolism',
    'map01250': 'Biosynthesis of nucleotide sugars',
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

# Drug-related pathways (will flag metabolites in these)
DRUG_PATHWAYS = [
    'map00980', 'map00982', 'map00983', 'map00232'  # xenobiotic/drug metabolism, caffeine
]


def is_drug_related(met_name: str) -> bool:
    """Check if metabolite is drug/xenobiotic related."""
    name_lower = met_name.lower()
    for pattern in DRUG_PATTERNS:
        if re.search(pattern, name_lower, re.IGNORECASE):
            return True
    return False


def get_pathway_name(pathway_id: str) -> str:
    """Get human-readable pathway name."""
    # Extract map ID from path:mapXXXXX format
    map_id = pathway_id.replace('path:', '')

    if map_id in KEGG_PATHWAY_NAMES:
        return KEGG_PATHWAY_NAMES[map_id]

    # Try to fetch from cache or API
    cache_file = Path(KEGG_CACHE_DIR) / f"pathway_{map_id}.json"
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
                if 'name' in data:
                    return data['name']
        except:
            pass

    return map_id  # Return ID if name not found


def fetch_pathway_genes(pathway_id: str) -> list:
    """Fetch genes/proteins associated with a pathway from KEGG."""
    map_id = pathway_id.replace('path:', '')
    cache_file = Path(KEGG_CACHE_DIR) / f"pathway_genes_{map_id}.json"

    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except:
            pass

    # Try KEGG API
    try:
        url = f"https://rest.kegg.jp/link/hsa/{map_id}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            genes = []
            for line in response.text.strip().split('\n'):
                if '\t' in line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        genes.append(parts[1])
            # Cache result
            Path(KEGG_CACHE_DIR).mkdir(exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump(genes, f)
            time.sleep(0.2)  # Rate limit
            return genes
    except:
        pass

    return []


def fetch_enzyme_genes(ec_number: str) -> list:
    """Fetch genes associated with an enzyme from KEGG."""
    ec_id = ec_number.replace('ec:', '')
    cache_file = Path(KEGG_CACHE_DIR) / f"enzyme_genes_{ec_id.replace('.', '_')}.json"

    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except:
            pass

    # Try KEGG API for human genes
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
            # Cache result
            Path(KEGG_CACHE_DIR).mkdir(exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump(genes, f)
            time.sleep(0.2)
            return genes
    except:
        pass

    return []


# =============================================================================
# STEP 1: Pathway-Level Aggregation
# =============================================================================

def aggregate_pathways(enzyme_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate enzyme scores to pathway level."""
    print("Step 1: Aggregating enzyme scores to pathway level...")

    pathway_data = defaultdict(lambda: {
        'enzymes': [],
        'enzyme_scores': [],
        'metabolites': set(),
        'total_score': 0.0
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

        for pathway in pathways:
            pathway_data[pathway]['enzymes'].append(ec_id)
            pathway_data[pathway]['enzyme_scores'].append(score)
            pathway_data[pathway]['metabolites'].update(mets)
            pathway_data[pathway]['total_score'] += score

    # Build dataframe
    rows = []
    for pathway_id, data in pathway_data.items():
        pathway_name = get_pathway_name(pathway_id)
        rows.append({
            'pathway_id': pathway_id,
            'pathway_name': pathway_name,
            'pathway_score': data['total_score'],
            'member_enzymes': '; '.join(data['enzymes']),
            'supporting_metabolites': '; '.join(data['metabolites']),
            'n_enzymes': len(data['enzymes']),
            'n_metabolites': len(data['metabolites'])
        })

    df = pd.DataFrame(rows)
    df = df.sort_values('pathway_score', ascending=False).reset_index(drop=True)

    print(f"  Found {len(df)} pathways")
    return df


# =============================================================================
# STEP 2: Expand to Proteins/Genes
# =============================================================================

def expand_to_proteins(pathway_df: pd.DataFrame) -> pd.DataFrame:
    """Add protein/gene information to pathways."""
    print("Step 2: Expanding pathways to proteins/genes...")

    member_proteins = []
    n_proteins = []

    for _, row in pathway_df.iterrows():
        enzymes = [e.strip() for e in str(row['member_enzymes']).split(';') if e.strip()]

        all_genes = set()
        for ec in enzymes[:5]:  # Limit API calls
            genes = fetch_enzyme_genes(ec)
            all_genes.update(genes)

        member_proteins.append('; '.join(sorted(all_genes)[:20]))  # Limit to top 20
        n_proteins.append(len(all_genes))

    pathway_df['member_proteins'] = member_proteins
    pathway_df['n_proteins'] = n_proteins

    print(f"  Added protein info for {len(pathway_df)} pathways")
    return pathway_df


# =============================================================================
# STEP 3: Drug/Xenobiotic Annotation
# =============================================================================

def annotate_metabolites(met_df: pd.DataFrame) -> pd.DataFrame:
    """Annotate metabolites as drug-related or endogenous."""
    print("Step 3: Annotating drug/xenobiotic metabolites...")

    met_df = met_df.copy()
    met_df['is_drug_related'] = met_df['metabolite'].apply(is_drug_related)

    n_drug = met_df['is_drug_related'].sum()
    n_endo = len(met_df) - n_drug
    print(f"  Drug-related: {n_drug}, Endogenous: {n_endo}")

    return met_df


def compute_endogenous_pathway_scores(enzyme_df: pd.DataFrame, met_df: pd.DataFrame) -> pd.DataFrame:
    """Recompute pathway scores excluding drug-related metabolites."""
    print("  Computing endogenous-only pathway scores...")

    # Get drug-related metabolites
    drug_mets = set(met_df[met_df['is_drug_related']]['metabolite'].str.lower())

    pathway_data = defaultdict(lambda: {
        'enzymes': [],
        'enzyme_scores': [],
        'metabolites': set(),
        'total_score': 0.0
    })

    for _, row in enzyme_df.iterrows():
        ec_id = row['enzyme_id']
        score = row['enzyme_score']

        # Parse metabolites and filter drug-related
        mets_str = str(row.get('supporting_metabolites', ''))
        mets = [m.strip() for m in mets_str.split(';') if m.strip()]
        endo_mets = [m for m in mets if m.lower() not in drug_mets]

        if not endo_mets:
            continue  # Skip enzyme if all metabolites are drug-related

        # Adjust score based on fraction of endogenous metabolites
        frac_endo = len(endo_mets) / len(mets) if mets else 0
        adjusted_score = score * frac_endo

        # Parse pathways
        pathways_str = str(row.get('pathways', ''))
        if not pathways_str or pathways_str == 'nan':
            continue

        pathways = [p.strip() for p in pathways_str.split(';') if p.strip()]

        for pathway in pathways:
            pathway_data[pathway]['enzymes'].append(ec_id)
            pathway_data[pathway]['enzyme_scores'].append(adjusted_score)
            pathway_data[pathway]['metabolites'].update(endo_mets)
            pathway_data[pathway]['total_score'] += adjusted_score

    # Build dataframe
    rows = []
    for pathway_id, data in pathway_data.items():
        pathway_name = get_pathway_name(pathway_id)
        rows.append({
            'pathway_id': pathway_id,
            'pathway_name': pathway_name,
            'pathway_score': data['total_score'],
            'member_enzymes': '; '.join(data['enzymes']),
            'supporting_metabolites': '; '.join(data['metabolites']),
            'n_enzymes': len(data['enzymes']),
            'n_metabolites': len(data['metabolites'])
        })

    df = pd.DataFrame(rows)
    df = df.sort_values('pathway_score', ascending=False).reset_index(drop=True)

    return df


# =============================================================================
# STEP 4: Top Pathway Subgraph Visualization
# =============================================================================

def visualize_top_pathway_subgraph(
    pathway_df: pd.DataFrame,
    enzyme_df: pd.DataFrame,
    met_df: pd.DataFrame,
    output_path: str
):
    """Create publication-ready visualization of top pathway."""
    print("Step 4: Creating top pathway subgraph visualization...")

    if len(pathway_df) == 0:
        print("  No pathways to visualize")
        return

    top_pathway = pathway_df.iloc[0]
    pathway_name = top_pathway['pathway_name']
    pathway_id = top_pathway['pathway_id']

    print(f"  Visualizing: {pathway_name}")

    # Get enzymes in this pathway
    pathway_enzymes = [e.strip() for e in str(top_pathway['member_enzymes']).split(';') if e.strip()]

    # Filter enzyme_df for this pathway
    enz_scores = {}
    enz_mets = {}
    for _, row in enzyme_df.iterrows():
        ec = row['enzyme_id']
        if ec in pathway_enzymes:
            enz_scores[ec] = row['enzyme_score']
            mets = [m.strip() for m in str(row['supporting_metabolites']).split(';') if m.strip()]
            enz_mets[ec] = mets

    # Sort enzymes by score, take top 10
    top_enzymes = sorted(enz_scores.keys(), key=lambda x: enz_scores.get(x, 0), reverse=True)[:10]

    # Collect supporting metabolites
    pathway_mets = set()
    for ec in top_enzymes:
        pathway_mets.update(enz_mets.get(ec, []))

    # Get metabolite importance scores
    met_scores = {}
    for _, row in met_df.iterrows():
        met_scores[row['metabolite']] = row['importance_score']

    # Sort metabolites by importance, take top 30
    top_mets = sorted(pathway_mets, key=lambda x: met_scores.get(x, 0), reverse=True)[:30]

    # Build graph
    G = nx.Graph()

    # Add enzyme nodes
    for ec in top_enzymes:
        G.add_node(ec, node_type='enzyme', score=enz_scores.get(ec, 0))

    # Add metabolite nodes
    for met in top_mets:
        G.add_node(met, node_type='metabolite', score=met_scores.get(met, 0))

    # Add edges (metabolite -> enzyme)
    for ec in top_enzymes:
        for met in enz_mets.get(ec, []):
            if met in top_mets:
                edge_weight = met_scores.get(met, 0) * enz_scores.get(ec, 0)
                G.add_edge(met, ec, weight=edge_weight)

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))

    # Separate nodes by type
    enz_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'enzyme']
    met_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'metabolite']

    # Position nodes in layers (left: metabolites, right: enzymes)
    pos = {}

    # Metabolites on left
    for i, met in enumerate(met_nodes):
        y = 1 - (i / max(len(met_nodes) - 1, 1))
        pos[met] = (0, y)

    # Enzymes on right
    for i, enz in enumerate(enz_nodes):
        y = 1 - (i / max(len(enz_nodes) - 1, 1))
        pos[enz] = (1, y)

    # Node sizes based on scores
    enz_sizes = [max(300, min(2000, G.nodes[n]['score'] * 1e8)) for n in enz_nodes]
    met_sizes = [max(200, min(1500, G.nodes[n]['score'] * 5000)) for n in met_nodes]

    # Draw edges
    edges = G.edges(data=True)
    if edges:
        edge_weights = [e[2].get('weight', 0.0001) for e in edges]
        max_weight = max(edge_weights) if edge_weights else 1
        edge_widths = [max(0.5, 3 * w / max_weight) for w in edge_weights]
        edge_alphas = [max(0.2, min(0.8, w / max_weight)) for w in edge_weights]

        for (u, v, d), width, alpha in zip(edges, edge_widths, edge_alphas):
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                   'gray', linewidth=width, alpha=alpha, zorder=1)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=met_nodes, node_color='#2ecc71',
                          node_size=met_sizes, alpha=0.8, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=enz_nodes, node_color='#e74c3c',
                          node_size=enz_sizes, alpha=0.8, ax=ax)

    # Labels
    met_labels = {n: n[:20] + '...' if len(n) > 20 else n for n in met_nodes}
    enz_labels = {n: n.replace('ec:', '') for n in enz_nodes}

    # Draw labels with offset
    for node, (x, y) in pos.items():
        if node in met_nodes:
            ax.text(x - 0.05, y, met_labels[node], fontsize=8, ha='right', va='center')
        else:
            ax.text(x + 0.05, y, enz_labels[node], fontsize=9, ha='left', va='center', fontweight='bold')

    # Legend
    legend_elements = [
        mpatches.Patch(color='#2ecc71', label='Metabolites'),
        mpatches.Patch(color='#e74c3c', label='Enzymes'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # Title
    ax.set_title(f'Top Pathway: {pathway_name}\n({pathway_id})', fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Saved: {output_path}")


# =============================================================================
# STEP 5: Summary Figures
# =============================================================================

def create_summary_figures(
    pathway_df: pd.DataFrame,
    pathway_endo_df: pd.DataFrame,
    enzyme_df: pd.DataFrame,
    met_df: pd.DataFrame,
    viz_dir: str
):
    """Generate summary visualizations."""
    print("Step 5: Creating summary figures...")

    # 1. Top 10 pathways bar chart
    plt.figure(figsize=(12, 8))
    top10 = pathway_endo_df.head(10).copy()
    top10['short_name'] = top10['pathway_name'].apply(lambda x: x[:40] + '...' if len(str(x)) > 40 else x)

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top10)))
    bars = plt.barh(range(len(top10)), top10['pathway_score'], color=colors)
    plt.yticks(range(len(top10)), top10['short_name'])
    plt.xlabel('Pathway Score (sum of enzyme importance)', fontsize=12)
    plt.title('Top 10 Pathways (Endogenous Metabolites Only)', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()

    # Add enzyme count annotations
    for i, (score, n_enz) in enumerate(zip(top10['pathway_score'], top10['n_enzymes'])):
        plt.text(score * 1.02, i, f'n={n_enz}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'top10_pathways_bar.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: top10_pathways_bar.png")

    # 2. Top 20 enzymes bar chart
    plt.figure(figsize=(12, 10))
    top20_enz = enzyme_df.head(20).copy()
    top20_enz['short_id'] = top20_enz['enzyme_id'].str.replace('ec:', '')

    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(top20_enz)))
    plt.barh(range(len(top20_enz)), top20_enz['enzyme_score'], color=colors)
    plt.yticks(range(len(top20_enz)), top20_enz['short_id'])
    plt.xlabel('Enzyme Importance Score', fontsize=12)
    plt.title('Top 20 Enzymes by Importance', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'top20_enzymes_bar.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: top20_enzymes_bar.png")

    # 3. Endogenous vs Drug metabolites
    plt.figure(figsize=(10, 6))

    drug_mets = met_df[met_df['is_drug_related']]
    endo_mets = met_df[~met_df['is_drug_related']]

    # Box plot
    data = [endo_mets['importance_score'].values, drug_mets['importance_score'].values]
    positions = [1, 2]

    bp = plt.boxplot(data, positions=positions, widths=0.6, patch_artist=True)
    bp['boxes'][0].set_facecolor('#2ecc71')
    bp['boxes'][1].set_facecolor('#e74c3c')

    plt.xticks([1, 2], [f'Endogenous\n(n={len(endo_mets)})', f'Drug-related\n(n={len(drug_mets)})'])
    plt.ylabel('Importance Score', fontsize=12)
    plt.title('Metabolite Importance: Endogenous vs Drug-Related', fontsize=14, fontweight='bold')

    # Add mean values
    for i, d in enumerate(data):
        if len(d) > 0:
            plt.text(positions[i], np.mean(d), f'μ={np.mean(d):.4f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'endogenous_vs_drug_metabolites.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: endogenous_vs_drug_metabolites.png")

    # 4. Pathway composition heatmap
    plt.figure(figsize=(14, 10))

    # Get top 15 pathways and their enzymes
    top_pathways = pathway_endo_df.head(15)

    # Build enzyme presence matrix
    all_enzymes = set()
    for _, row in top_pathways.iterrows():
        enzymes = [e.strip() for e in str(row['member_enzymes']).split(';') if e.strip()]
        all_enzymes.update(enzymes)

    # Get top 30 enzymes by score
    top_enz_list = enzyme_df.head(30)['enzyme_id'].tolist()
    relevant_enzymes = [e for e in top_enz_list if e in all_enzymes][:20]

    if relevant_enzymes:
        # Build matrix
        matrix = np.zeros((len(top_pathways), len(relevant_enzymes)))

        for i, (_, row) in enumerate(top_pathways.iterrows()):
            enzymes = [e.strip() for e in str(row['member_enzymes']).split(';') if e.strip()]
            for j, enz in enumerate(relevant_enzymes):
                if enz in enzymes:
                    # Get enzyme score
                    enz_score = enzyme_df[enzyme_df['enzyme_id'] == enz]['enzyme_score'].values
                    matrix[i, j] = enz_score[0] if len(enz_score) > 0 else 0

        # Plot heatmap
        pathway_names = [str(n)[:35] + '...' if len(str(n)) > 35 else str(n)
                        for n in top_pathways['pathway_name']]
        enzyme_names = [e.replace('ec:', '') for e in relevant_enzymes]

        sns.heatmap(matrix, xticklabels=enzyme_names, yticklabels=pathway_names,
                   cmap='YlOrRd', annot=False, fmt='.2e')
        plt.xlabel('Enzymes', fontsize=12)
        plt.ylabel('Pathways', fontsize=12)
        plt.title('Pathway-Enzyme Composition\n(Cell value = enzyme importance score)',
                 fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'pathway_composition_heatmap.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: pathway_composition_heatmap.png")


# =============================================================================
# STEP 6: Text Summary
# =============================================================================

def write_summary(
    pathway_df: pd.DataFrame,
    pathway_endo_df: pd.DataFrame,
    enzyme_df: pd.DataFrame,
    met_df: pd.DataFrame,
    output_path: str
):
    """Generate plain-language summary for collaborators."""
    print("Step 6: Writing summary...")

    n_drug = met_df['is_drug_related'].sum()
    n_endo = len(met_df) - n_drug

    top_pathway = pathway_endo_df.iloc[0] if len(pathway_endo_df) > 0 else None

    summary = f"""
================================================================================
PATHWAY-LEVEL ANALYSIS SUMMARY
================================================================================

METHODOLOGY
-----------
This analysis derives pathway-level importance from a supervised heterogeneous
graph neural network (HeteroGNN) trained to classify pain levels (High vs Low).

Key points:
1. The GNN was trained using metabolite abundance data as edge weights
2. Enzyme importance scores were computed via gradient-based attribution
3. Pathways are ranked by aggregating their member enzyme scores
4. GROUP labels (disease subtypes) were NOT used as model features
   - This ensures the model learns pain-specific patterns, not group confounds

PATHWAY DERIVATION
------------------
- Pathway scores = sum of importance scores for all enzymes in the pathway
- Each pathway contains multiple enzymes (EC numbers)
- Each enzyme may correspond to multiple genes/proteins

DRUG vs ENDOGENOUS METABOLITES
------------------------------
We separately report results with and without drug-related metabolites:
- Total metabolites analyzed: {len(met_df)}
- Drug/xenobiotic-related: {n_drug} ({100*n_drug/len(met_df):.1f}%)
- Endogenous: {n_endo} ({100*n_endo/len(met_df):.1f}%)

Rationale: Drug metabolites (e.g., metformin, diazepam) reflect medication use
rather than intrinsic pain biology. Separating these helps identify endogenous
pathways relevant to pain mechanisms.

TOP FINDINGS (Endogenous Only)
------------------------------
"""

    if top_pathway is not None:
        summary += f"""
Top Pathway: {top_pathway['pathway_name']}
  - Pathway ID: {top_pathway['pathway_id']}
  - Score: {top_pathway['pathway_score']:.2e}
  - Number of enzymes: {top_pathway['n_enzymes']}
  - Number of metabolites: {top_pathway['n_metabolites']}

Top 5 Pathways:
"""
        for i, (_, row) in enumerate(pathway_endo_df.head(5).iterrows()):
            summary += f"  {i+1}. {row['pathway_name']} (score={row['pathway_score']:.2e}, n_enzymes={row['n_enzymes']})\n"

    summary += f"""
Top 5 Enzymes:
"""
    for i, (_, row) in enumerate(enzyme_df.head(5).iterrows()):
        summary += f"  {i+1}. {row['enzyme_id']} (score={row['enzyme_score']:.2e})\n"

    summary += f"""
INTERPRETATION NOTES
--------------------
- Higher pathway scores indicate stronger association with pain classification
- Pathways like "Metabolic pathways" (map01100) are very broad umbrella categories
- More specific pathways (e.g., "Purine metabolism", "Bile acid biosynthesis")
  may be more interpretable for follow-up experiments

OUTPUT FILES
------------
- pathway_rank.csv: All pathways ranked by enzyme score sum
- pathway_rank_endogenous.csv: Excluding drug-related metabolites
- pathway_rank_all.csv: Including all metabolites
- gnn_viz/top_pathway_subgraph.png: Visualization of top pathway
- gnn_viz/top10_pathways_bar.png: Bar chart of top 10 pathways
- gnn_viz/top20_enzymes_bar.png: Bar chart of top 20 enzymes

================================================================================
Generated by pathway_analysis.py
================================================================================
"""

    with open(output_path, 'w') as f:
        f.write(summary)

    print(f"  Saved: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("PATHWAY-LEVEL AGGREGATION AND VISUALIZATION")
    print("=" * 60)

    # Load data
    enzyme_df = pd.read_csv(os.path.join(OUT_DIR, 'enzyme_rank.csv'))
    met_df = pd.read_csv(os.path.join(OUT_DIR, 'metabolite_rank.csv'))

    print(f"Loaded {len(enzyme_df)} enzymes, {len(met_df)} metabolites")

    # Ensure viz directory exists
    os.makedirs(VIZ_DIR, exist_ok=True)

    # Step 1: Pathway aggregation
    pathway_df = aggregate_pathways(enzyme_df)

    # Step 2: Expand to proteins (limited API calls)
    pathway_df = expand_to_proteins(pathway_df)

    # Step 3: Annotate metabolites
    met_df = annotate_metabolites(met_df)
    met_df.to_csv(os.path.join(OUT_DIR, 'metabolite_rank_annotated.csv'), index=False)

    # Compute endogenous-only pathway scores
    pathway_endo_df = compute_endogenous_pathway_scores(enzyme_df, met_df)
    pathway_endo_df = expand_to_proteins(pathway_endo_df)

    # Save pathway files
    pathway_df.to_csv(os.path.join(OUT_DIR, 'pathway_rank.csv'), index=False)
    pathway_df.to_csv(os.path.join(OUT_DIR, 'pathway_rank_all.csv'), index=False)
    pathway_endo_df.to_csv(os.path.join(OUT_DIR, 'pathway_rank_endogenous.csv'), index=False)

    print(f"Saved pathway_rank.csv ({len(pathway_df)} pathways)")
    print(f"Saved pathway_rank_endogenous.csv ({len(pathway_endo_df)} pathways)")

    # Step 4: Top pathway visualization
    visualize_top_pathway_subgraph(
        pathway_endo_df, enzyme_df, met_df,
        os.path.join(VIZ_DIR, 'top_pathway_subgraph.png')
    )

    # Step 5: Summary figures
    create_summary_figures(pathway_df, pathway_endo_df, enzyme_df, met_df, VIZ_DIR)

    # Step 6: Text summary
    write_summary(pathway_df, pathway_endo_df, enzyme_df, met_df,
                 os.path.join(OUT_DIR, 'path_summary.txt'))

    print("=" * 60)
    print("PATHWAY ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
