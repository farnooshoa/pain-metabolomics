# Pain Heterogeneous GNN Pipeline

End-to-end supervised heterogeneous graph neural network pipeline for pain classification using metabolomics data.

## Overview

This pipeline:
1. **LLM Name Normalization**: Uses OpenRouter (google/gemini-2.5-flash) to normalize metabolite names and generate KEGG query candidates
2. **KEGG REST API Mapping**: Maps metabolites to KEGG compounds, reactions, enzymes, and pathways
3. **Heterogeneous Graph Construction**: Builds a knowledge-guided graph with sample, metabolite, and enzyme nodes
4. **HeteroGNN Training**: Trains a supervised heterogeneous GNN for pain classification (High vs Low)
5. **Enzyme Ranking**: Outputs wet-lab-ready ranked enzyme list based on gradient attribution

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

### OpenRouter API Key

**IMPORTANT**: Set your OpenRouter API key as an environment variable:

```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

The pipeline will fail with a clear error message if this is not set. API keys are never hardcoded or logged.

## Usage

### Basic Usage

```bash
python pain_hetero_pipeline.py --csv starting_template.csv --outdir results
```

### With Supplementary KEGG Excel

```bash
python pain_hetero_pipeline.py --csv starting_template.csv --kegg-excel "KEGG_Ruiheng.xlsx" --outdir results
```

### Full Options

```bash
python pain_hetero_pipeline.py \
    --csv starting_template.csv \
    --outdir results \
    --kegg-excel "KEGG_Ruiheng.xlsx" \
    --use-metmet-prior \
    --k-metmet 10 \
    --run-permutation \
    --n-permutations 200 \
    --seed 42
```

### Skip LLM Calls (Use Cache)

If you have already run the pipeline and have `aliases_cache.json`, you can skip LLM calls:

```bash
python pain_hetero_pipeline.py --csv starting_template.csv --outdir results --disable-llm
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--csv` | (required) | Path to metabolomics CSV/Excel file |
| `--outdir` | `out_hetero_gnn` | Output directory |
| `--kegg-excel` | None | Optional supplementary KEGG Excel file |
| `--high-threshold` | 7 | PAIN_SCORE >= this is High pain |
| `--low-threshold` | 3 | PAIN_SCORE <= this is Low pain |
| `--use-metmet-prior` | True | Use metabolite-metabolite pathway edges |
| `--no-metmet-prior` | - | Disable met-met edges |
| `--k-metmet` | 10 | Max neighbors for met-met edges |
| `--hidden-dim` | 64 | GNN hidden dimension |
| `--n-layers` | 2 | Number of GNN layers |
| `--dropout` | 0.1 | Dropout rate |
| `--epochs` | 200 | Max training epochs |
| `--lr` | 0.001 | Learning rate |
| `--patience` | 15 | Early stopping patience |
| `--disable-llm` | False | Use cached aliases only |
| `--run-permutation` | False | Run permutation test |
| `--n-permutations` | 200 | Number of permutations |
| `--seed` | 42 | Random seed |

## Input Data Format

The input CSV/Excel file should contain:

- `CLIENT_SAMPLE_ID`: Unique sample identifier
- `PAIN_SCORE`: Numeric pain score (0-10)
- `Age`: (Optional) Patient age
- `GROUP_NAME1` / `GROUP_NAME2`: (Optional) Group labels for diagnostics
- Severity column: Values among {Severe, Moderate, Mild, Unknown}
- Many metabolite columns (numeric)

**IMPORTANT**: GROUP columns are **NOT** used as model features. They are only used for diagnostic analysis (confounding checks, within-group AUC).

## Output Files

### Main Results

| File | Description |
|------|-------------|
| `enzyme_rank.csv` | Ranked enzymes with scores, supporting metabolites, pathways |
| `metabolite_rank.csv` | Ranked metabolites with importance scores |
| `cv_metrics_overall.csv` | Per-fold AUC/Accuracy + mean/std |
| `cv_metrics_by_group.csv` | AUC within each group subset |
| `mapping_report.csv` | LLM queries, KEGG mappings, confidence scores |

### KEGG Mappings

| File | Description |
|------|-------------|
| `met2cpd.json` | Metabolite -> KEGG compound mapping |
| `met2rxn.json` | Metabolite -> reactions mapping |
| `met2pathway.json` | Metabolite -> pathways mapping |
| `met2ec.json` | Metabolite -> enzyme (EC) mapping |

### Diagnostics

| File | Description |
|------|-------------|
| `group_vs_label_table.csv` | Contingency table of group vs pain label |
| `permutation_auc_distribution.csv` | AUC distribution from permutation test |
| `sanity_check_results.csv` | Results of sanity checks |

### Visualizations (in `gnn_viz/`)

| File | Description |
|------|-------------|
| `roc_overall.png` | ROC curves for all folds |
| `confusion_matrix.png` | Confusion matrix |
| `group_vs_label_heatmap.png` | Group vs pain label distribution |
| `permutation_auc_hist.png` | Permutation test histogram |
| `metabolite_stability.png` | Top metabolites by selection frequency |
| `enzyme_stability.png` | Top enzymes by selection frequency |
| `auc_by_group.png` | AUC for each group |

## Graph Structure

The pipeline builds a heterogeneous graph with:

### Node Types
- **sample**: One per patient sample
  - Features: [Age, Severity one-hot] (GROUP is NOT included!)
- **met**: One per metabolite
  - Features: Constant vector (to avoid leakage)
- **enz**: One per enzyme (EC number)
  - Features: Constant vector

### Edge Types
1. **sample -> met** (`has`): Bipartite edges with `edge_attr = abundance`
2. **met -> sample** (`rev_has`): Reverse edges with same `edge_attr`
3. **met -> enz** (`to`): From KEGG mapping
4. **enz -> met** (`to`): Reverse enzyme edges
5. **met -> met** (`pathway`): Metabolites sharing KEGG pathways (optional, sparsified)

## Enzyme Ranking Method

1. **Metabolite Importance**: Computed via gradient attribution on sample->met edges:
   ```
   importance(m) = mean_s |grad(logit_s) * edge_attr(s,m)|
   ```

2. **Enzyme Score**: Propagated from metabolite importance via met->enz mapping:
   ```
   enzyme_score(ec) = sum_{m linked to ec} importance(m)
   ```

3. **Aggregation**: Scores are averaged across CV folds, with selection frequency tracked.

## Caching

- **LLM Aliases**: Cached in `aliases_cache.json` (reuse with `--disable-llm`)
- **KEGG Responses**: Cached in `kegg_cache/` directory (URL-hashed filenames)

Caching ensures:
- Reproducibility across runs
- Reduced API costs
- Faster re-runs

## Notes

1. **Group Confounding**: If GROUP perfectly predicts pain (e.g., HZ_H = all high pain), the model may learn group patterns. Check `group_vs_label_heatmap.png` and Cramer's V in logs.

2. **Permutation Test**: Provides statistical significance of the model's AUC. P-value < 0.05 indicates the model performs significantly better than chance.

3. **Sanity Checks**: Shuffling `edge_attr` should decrease AUC, confirming the model uses metabolite abundances meaningfully.

## License

MIT License
