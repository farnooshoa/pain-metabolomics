# Pain Metabolomics Heterogeneous GNN

Heterogeneous graph neural network pipeline for pain classification from metabolomics data, with KEGG-based enzyme/pathway integration and a comprehensive XAI suite.

**Cohort**: Herpes Zoster (HZ_HL, n=24) + Post-Herpetic Neuralgia (PHN, n=12). Binary task: High Pain (Pain Score ≥4) vs Low Pain (≤3).

---

## Quick start

```bash
# Activate environment (or install requirements.txt)
conda activate flashtalk          # or your equivalent env with torch + torch-geometric

# Place patient data file (NOT in repo) at project root
# starting_template.csv  ← gitignored, request from PI

# Run the main pipeline (n=36, 1019 endogenous metabolites, 6-fold CV)
cd pain_hetero_pipeline
OPENROUTER_API_KEY=dummy python pain_hetero_pipeline.py \
    --csv ../starting_template.csv \
    --outdir out_hetero_endogenous_6fold \
    --high-threshold 4 --low-threshold 3 \
    --disable-llm --exclude-xenobiotics --n-folds 6 --seed 42
```

---

## Headline results (v2.0, 2026-04)

| Model | AUC | F1 | Notes |
|-------|-----|----|----|
| **Main (1019 mets, 6-fold CV)** | **0.796 ± 0.123** | 0.701 | `out_hetero_endogenous_6fold/` |
| Top-250 differed (Wilcoxon-prefilter) | **0.838 ± 0.141** | 0.766 | `out_hetero_top250/` |
| L1-sparse | 0.721 | — | `out_hetero_endogenous_6fold_L1/` |

- **Overall permutation test**: Z = +4.59, p < 0.005 (model is significantly better than label-shuffled null)
- **Per-feature FDR (q<0.05)**: 0/1019 metabolites, 0/966 enzymes — signal is multivariate, not driven by single biomarkers
- **Convergent biological themes**: bile acid metabolism, purine/xanthine metabolism, neurosteroids (DHEA-S, androgen sulfates), gamma-glutamyl/glutathione

For full analysis see [`ANALYSIS_REPORT.md`](ANALYSIS_REPORT.md). For onboarding see [`HANDOVER.md`](HANDOVER.md).

---

## Repository structure

```
.
├── README.md                    # this file
├── HANDOVER.md                  # detailed onboarding for next person
├── ANALYSIS_REPORT.md           # full v2.0 report (7 sections)
├── EMAIL_DRAFT_*.md             # collaborator correspondence drafts
├── attachments_for_collaborator_*/  # files sent to wet-lab collaborator
├── metabolites names and pathways.xlsx    # Metabolon annotation reference
└── pain_hetero_pipeline/
    ├── pain_hetero_pipeline.py  # main pipeline
    ├── utils.py                 # data loading + preprocessing
    ├── kegg_mapper.py           # KEGG mapping
    ├── openrouter_aliases.py    # LLM-based name normalization
    ├── permutation_test_*.py    # statistical validation (Methods A–B)
    ├── l1_sparse_pipeline.py    # L1-sparse variant (Method C)
    ├── subsample_power_analysis.py     # power vs n (Method D)
    ├── xai_*.py                 # XAI suite (baselines, ablation, LOSO,
    │                              confounders, integrated gradients)
    ├── rescore_enzymes.py       # 6 alternative enzyme scoring formulas
    ├── run_top250_differed.py   # filter to most-differed metabolites
    └── out_hetero_*/            # per-experiment outputs (CSVs, HTML)
```

---

## Data files (NOT in repo)

The following are gitignored due to patient privacy / IRB:

- `starting_template.csv` — main patient data (48 samples × 1592 metabolites). Request from PI.
- `KEGG_Ruiheng.xlsx`, `L8000_*.xlsx` — older raw files

`metabolites names and pathways.xlsx` (Metabolon annotation, no patient data) is included.

---

## Statistical / XAI methods (see ANALYSIS_REPORT.md §5–§7)

- **Method A**: per-feature label-permutation FDR
- **Method B**: pathway-aggregated permutation
- **Method C**: L1-sparse GNN with permutation
- **Method D**: subsample power analysis
- **§7.1** Baseline comparison (LogReg, RandomForest)
- **§7.2** Univariate Wilcoxon audit
- **§7.3** Top-K ablation (vs random-K control)
- **§7.4** Leave-one-sample-out stability
- **§7.5** Confounder analysis (Pain vs Group / Severity / Age)
- **§7.6** Integrated Gradients per-sample attribution
- **§7.8** Enzyme degree-bias diagnostic + 6 alternative scorings
- **§7.9** Top-250 differed-metabolite model

---

## Open issues / next steps

1. **Confounder concern**: top-30 enzyme list is identical when training on Pain vs Group / Severity / Age (XAI §7.5). Need covariate-adjusted models.
2. **Degree bias**: SUM enzyme score correlates r=0.74 with degree (XAI §7.8). Use MEAN/TOP5_MEAN or specific-only filter.
3. **Sample size**: n=36 limits per-feature FDR power. Power analysis suggests n≈70 for stable single-biomarker claims.
4. **Per-feature significance**: 0 metabolites/enzymes survive FDR. Frame as multivariate signature.

---

## Environment

- Python 3.10
- `torch`, `torch-geometric`, `pandas`, `numpy`, `scikit-learn`, `scipy`, `tqdm`, `openpyxl`
- See `pain_hetero_pipeline/requirements.txt`
- Tested on CUDA (4× GPUs available locally)

---

## License

[TBD — internal lab use until publication]
