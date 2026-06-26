"""
Utility functions for the Pain Hetero GNN Pipeline.
"""

import re
import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from scipy import stats
from collections import Counter

logger = logging.getLogger(__name__)

def _load_xenobiotic_set(annotation_path: str = 'metabolites names and pathways.xlsx') -> set:
    """Load xenobiotic metabolite names from Metabolon annotation file."""
    p = Path(annotation_path)
    if not p.exists():
        logger.warning(f"Annotation file not found: {p}. Cannot filter xenobiotics.")
        return set()
    annot = pd.read_excel(p, sheet_name='Chemical Annotation')
    xeno = annot.loc[annot['SUPER_PATHWAY'] == 'Xenobiotics', 'CHEMICAL_NAME']
    return set(xeno.values)


def load_metabolomics_data(
    csv_path: str,
    high_threshold: int = 7,
    low_threshold: int = 3,
    exclude_xenobiotics: bool = False,
    annotation_path: str = 'metabolites names and pathways.xlsx'
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
    """
    Load and preprocess metabolomics data.

    Args:
        csv_path: Path to the CSV/Excel file
        high_threshold: PAIN_SCORE >= this is High pain
        low_threshold: PAIN_SCORE <= this is Low pain
        exclude_xenobiotics: If True, remove xenobiotic/drug metabolites
        annotation_path: Path to Metabolon annotation file with SUPER_PATHWAY

    Returns:
        Tuple of (metabolite_df, labels, group_labels, metabolite_columns, metadata)
    """
    # Try reading as Excel first, then CSV
    try:
        df = pd.read_excel(csv_path)
        logger.info(f"Loaded Excel file: {csv_path}")
    except Exception:
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded CSV file: {csv_path}")
        except Exception as e:
            raise ValueError(f"Could not load data file: {e}")

    logger.info(f"Raw data shape: {df.shape}")

    # Find pain score column
    pain_col = None
    for col in df.columns:
        if 'PAIN' in col.upper() and 'SCORE' in col.upper():
            pain_col = col
            break
    if not pain_col:
        raise ValueError("Could not find PAIN_SCORE column")

    # Filter to High and Low pain samples
    df_filtered = df[
        (df[pain_col] >= high_threshold) | (df[pain_col] <= low_threshold)
    ].copy()

    # Create binary labels
    labels = (df_filtered[pain_col] >= high_threshold).astype(int).values
    logger.info(f"Filtered samples: {len(df_filtered)} (High: {labels.sum()}, Low: {len(labels) - labels.sum()})")

    # Find group column (for diagnostics only)
    group_col = None
    for col in df.columns:
        if 'GROUP' in col.upper():
            group_col = col
            break

    group_labels = None
    if group_col:
        group_labels = df_filtered[group_col].values
        logger.info(f"Groups found: {Counter(group_labels)}")

    # Find severity column
    severity_col = None
    severity_values = {'Severe', 'Moderate', 'Mild', 'Unknown'}
    for col in df.columns:
        unique_vals = set(df[col].dropna().astype(str).unique())
        if unique_vals & severity_values:
            severity_col = col
            break

    # Find age column
    age_col = None
    for col in df.columns:
        if 'AGE' in col.upper():
            age_col = col
            break

    # Identify metabolite columns (numeric, not metadata)
    exclude_cols = {
        pain_col, group_col, severity_col, age_col,
        'CLIENT_SAMPLE_ID', 'SAMPLE_ID', 'ID'
    }
    exclude_cols = {c for c in exclude_cols if c}  # Remove None

    # Also exclude columns that are clearly not metabolites
    metabolite_cols = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        if col.upper() in ['GROUP_NAME1', 'GROUP_NAME2', 'UNNAMED']:
            continue
        if 'UNNAMED' in col.upper():
            continue
        if 'GROUP' in col.upper():
            continue
        # Exclude any pain-score-like columns (e.g. "Pain Score", "PAIN_SCORE")
        if 'PAIN' in col.upper() and 'SCORE' in col.upper():
            logger.info(f"Excluding pain score column from features: '{col}'")
            continue

        # Check if numeric
        try:
            vals = pd.to_numeric(df_filtered[col], errors='coerce')
            if vals.notna().sum() > len(df_filtered) * 0.3:  # At least 30% non-null
                metabolite_cols.append(col)
        except Exception:
            pass

    logger.info(f"Identified {len(metabolite_cols)} metabolite columns")

    # Optionally exclude xenobiotic/drug metabolites
    if exclude_xenobiotics:
        xeno_set = _load_xenobiotic_set(annotation_path)
        xeno_cols = [c for c in metabolite_cols if c in xeno_set]
        metabolite_cols = [c for c in metabolite_cols if c not in xeno_set]
        logger.info(f"Excluded {len(xeno_cols)} xenobiotic metabolites, {len(metabolite_cols)} endogenous remaining")
        if xeno_cols:
            logger.info(f"Excluded xenobiotics (first 10): {xeno_cols[:10]}")
        # Also exclude unidentified X- metabolites
        x_cols = [c for c in metabolite_cols if c.startswith('X-')]
        metabolite_cols = [c for c in metabolite_cols if not c.startswith('X-')]
        logger.info(f"Excluded {len(x_cols)} unidentified X- metabolites, {len(metabolite_cols)} identified endogenous remaining")

    # Extract metabolite data
    metabolite_df = df_filtered[metabolite_cols].copy()

    # Extract metadata
    metadata = {
        'sample_ids': df_filtered['CLIENT_SAMPLE_ID'].values if 'CLIENT_SAMPLE_ID' in df.columns else np.arange(len(df_filtered)),
        'pain_scores': df_filtered[pain_col].values,
        'severity_col': severity_col,
        'age_col': age_col,
        'group_col': group_col
    }

    if severity_col:
        metadata['severity'] = df_filtered[severity_col].values
    if age_col:
        metadata['age'] = df_filtered[age_col].values

    return metabolite_df, labels, group_labels, metabolite_cols, metadata


def preprocess_fold(
    X_train: np.ndarray,
    X_val: np.ndarray,
    age_train: Optional[np.ndarray] = None,
    age_val: Optional[np.ndarray] = None,
    severity_train: Optional[np.ndarray] = None,
    severity_val: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess data within a CV fold (imputation + standardization).

    Args:
        X_train: Training metabolite data
        X_val: Validation metabolite data
        age_train, age_val: Age values
        severity_train, severity_val: Severity categories

    Returns:
        Tuple of (X_train_scaled, X_val_scaled, sample_features_train, sample_features_val)
    """
    # Impute missing values with training median
    medians = np.nanmedian(X_train, axis=0)
    medians = np.nan_to_num(medians, nan=0.0)

    X_train_imp = X_train.copy()
    X_val_imp = X_val.copy()

    for j in range(X_train.shape[1]):
        X_train_imp[np.isnan(X_train_imp[:, j]), j] = medians[j]
        X_val_imp[np.isnan(X_val_imp[:, j]), j] = medians[j]

    # Standardize using training statistics
    means = X_train_imp.mean(axis=0)
    stds = X_train_imp.std(axis=0)
    stds[stds == 0] = 1.0

    X_train_scaled = (X_train_imp - means) / stds
    X_val_scaled = (X_val_imp - means) / stds

    # Build sample features (Age + Severity one-hot)
    sample_features_train = build_sample_features(age_train, severity_train)
    sample_features_val = build_sample_features(age_val, severity_val)

    # Scale age using training stats
    if age_train is not None:
        age_mean = np.nanmean(age_train)
        age_std = np.nanstd(age_train)
        if age_std == 0:
            age_std = 1.0

        # Age is first column
        sample_features_train[:, 0] = (sample_features_train[:, 0] - age_mean) / age_std
        sample_features_val[:, 0] = (sample_features_val[:, 0] - age_mean) / age_std

    return X_train_scaled, X_val_scaled, sample_features_train, sample_features_val


def build_sample_features(
    age: Optional[np.ndarray],
    severity: Optional[np.ndarray]
) -> np.ndarray:
    """
    Build sample node features from age and severity.

    Args:
        age: Age values
        severity: Severity categories

    Returns:
        Feature matrix [n_samples, n_features]
    """
    n_samples = len(age) if age is not None else len(severity) if severity is not None else 0
    if n_samples == 0:
        return np.zeros((0, 5))

    features = []

    # Age feature (will be scaled later)
    if age is not None:
        age_feat = np.array(age, dtype=float)
        age_feat = np.nan_to_num(age_feat, nan=np.nanmean(age_feat))
        features.append(age_feat.reshape(-1, 1))
    else:
        features.append(np.zeros((n_samples, 1)))

    # Severity one-hot (Severe, Moderate, Mild, Unknown)
    severity_cats = ['Severe', 'Moderate', 'Mild', 'Unknown']
    if severity is not None:
        severity_onehot = np.zeros((n_samples, len(severity_cats)))
        for i, sev in enumerate(severity):
            sev_str = str(sev) if pd.notna(sev) else 'Unknown'
            for j, cat in enumerate(severity_cats):
                if cat.lower() in sev_str.lower():
                    severity_onehot[i, j] = 1.0
                    break
            else:
                severity_onehot[i, -1] = 1.0  # Unknown
        features.append(severity_onehot)
    else:
        # Default to Unknown
        severity_onehot = np.zeros((n_samples, len(severity_cats)))
        severity_onehot[:, -1] = 1.0
        features.append(severity_onehot)

    return np.hstack(features)


def compute_group_confounding(
    group_labels: np.ndarray,
    pain_labels: np.ndarray
) -> Dict[str, Any]:
    """
    Compute group-label confounding statistics.

    Args:
        group_labels: Group assignments
        pain_labels: Binary pain labels

    Returns:
        Dict with chi2, p-value, cramers_v, contingency_table
    """
    from scipy.stats import chi2_contingency

    # Create contingency table
    groups = sorted(set(group_labels))
    table = np.zeros((len(groups), 2), dtype=int)

    group_idx = {g: i for i, g in enumerate(groups)}
    for g, y in zip(group_labels, pain_labels):
        table[group_idx[g], int(y)] += 1

    # Chi-square test
    try:
        chi2, p_value, dof, expected = chi2_contingency(table)

        # Cramer's V
        n = table.sum()
        min_dim = min(table.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0.0
    except Exception:
        chi2, p_value, cramers_v = 0.0, 1.0, 0.0

    return {
        'chi2': chi2,
        'p_value': p_value,
        'cramers_v': cramers_v,
        'contingency_table': table,
        'groups': groups
    }


def compute_within_group_auc(
    predictions: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray
) -> Dict[str, float]:
    """
    Compute AUC within each group.

    Args:
        predictions: Predicted probabilities
        labels: True binary labels
        groups: Group assignments

    Returns:
        Dict mapping group name to AUC (or None if both classes not present)
    """
    from sklearn.metrics import roc_auc_score

    results = {}
    for group in sorted(set(groups)):
        mask = groups == group
        y_true = labels[mask]
        y_pred = predictions[mask]

        if len(set(y_true)) < 2:
            results[str(group)] = None
        else:
            try:
                results[str(group)] = roc_auc_score(y_true, y_pred)
            except Exception:
                results[str(group)] = None

    return results


def create_visualizations(
    output_dir: str,
    cv_results: Dict[str, Any],
    metabolite_stability: Dict[str, int],
    enzyme_stability: Dict[str, int],
    group_stats: Optional[Dict] = None,
    permutation_aucs: Optional[List[float]] = None,
    real_auc: Optional[float] = None
):
    """
    Create all visualization plots.

    Args:
        output_dir: Output directory for plots
        cv_results: Cross-validation results
        metabolite_stability: Metabolite selection frequency
        enzyme_stability: Enzyme selection frequency
        group_stats: Group confounding statistics
        permutation_aucs: Permutation test AUCs
        real_auc: Real model AUC
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay

    viz_dir = Path(output_dir) / "gnn_viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # 1. ROC Curve
    if 'fold_predictions' in cv_results:
        plt.figure(figsize=(8, 6))
        for fold_idx, (y_true, y_pred) in enumerate(cv_results['fold_predictions']):
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            auc = cv_results['fold_aucs'][fold_idx]
            plt.plot(fpr, tpr, alpha=0.5, label=f'Fold {fold_idx+1} (AUC={auc:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves (Mean AUC={np.mean(cv_results["fold_aucs"]):.3f})')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(viz_dir / 'roc_overall.png', dpi=150)
        plt.close()

    # 2. Confusion Matrix
    if 'all_predictions' in cv_results:
        y_true_all = np.concatenate([y for y, _ in cv_results['all_predictions']])
        y_pred_all = np.concatenate([p for _, p in cv_results['all_predictions']])
        y_pred_binary = (y_pred_all >= 0.5).astype(int)

        cm = confusion_matrix(y_true_all, y_pred_binary)
        plt.figure(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(cm, display_labels=['Low Pain', 'High Pain'])
        disp.plot(cmap='Blues')
        plt.title('Confusion Matrix (All Folds)')
        plt.tight_layout()
        plt.savefig(viz_dir / 'confusion_matrix.png', dpi=150)
        plt.close()

    # 3. Group vs Label Heatmap
    if group_stats and 'contingency_table' in group_stats:
        plt.figure(figsize=(8, 5))
        table = group_stats['contingency_table']
        groups = group_stats['groups']

        # Normalize by row
        row_sums = table.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        table_norm = table / row_sums

        plt.imshow(table_norm, cmap='RdYlBu_r', aspect='auto')
        plt.colorbar(label='Proportion')
        plt.xticks([0, 1], ['Low Pain', 'High Pain'])
        plt.yticks(range(len(groups)), groups)
        plt.xlabel('Pain Label')
        plt.ylabel('Group')

        # Add counts
        for i in range(len(groups)):
            for j in range(2):
                plt.text(j, i, f'{table[i, j]}', ha='center', va='center',
                        color='white' if table_norm[i, j] > 0.5 else 'black')

        plt.title(f"Group vs Label (Cramer's V = {group_stats['cramers_v']:.3f})")
        plt.tight_layout()
        plt.savefig(viz_dir / 'group_vs_label_heatmap.png', dpi=150)
        plt.close()

    # 4. Permutation Histogram
    if permutation_aucs and real_auc is not None:
        plt.figure(figsize=(8, 5))
        plt.hist(permutation_aucs, bins=30, alpha=0.7, color='gray', edgecolor='black')
        plt.axvline(real_auc, color='red', linewidth=2, label=f'Real AUC = {real_auc:.3f}')

        p_value = np.mean([p >= real_auc for p in permutation_aucs])
        plt.xlabel('AUC')
        plt.ylabel('Count')
        plt.title(f'Permutation Test (p-value = {p_value:.4f})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(viz_dir / 'permutation_auc_hist.png', dpi=150)
        plt.close()

    # 5. Metabolite Stability
    if metabolite_stability:
        top_mets = sorted(metabolite_stability.items(), key=lambda x: -x[1])[:20]
        if top_mets:
            mets, counts = zip(*top_mets)

            plt.figure(figsize=(10, 6))
            plt.barh(range(len(mets)), counts, color='steelblue')
            plt.yticks(range(len(mets)), [m[:40] for m in mets])
            plt.xlabel('Selection Frequency (across folds)')
            plt.title('Top 20 Metabolites by Stability')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(viz_dir / 'metabolite_stability.png', dpi=150)
            plt.close()

    # 6. Enzyme Stability
    if enzyme_stability:
        top_enz = sorted(enzyme_stability.items(), key=lambda x: -x[1])[:10]
        if top_enz:
            enzymes, counts = zip(*top_enz)

            plt.figure(figsize=(10, 5))
            plt.barh(range(len(enzymes)), counts, color='darkgreen')
            plt.yticks(range(len(enzymes)), enzymes)
            plt.xlabel('Selection Frequency (across folds)')
            plt.title('Top 10 Enzymes by Stability')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(viz_dir / 'enzyme_stability.png', dpi=150)
            plt.close()

    # 7. AUC by Group
    if 'group_aucs' in cv_results:
        group_aucs = cv_results['group_aucs']
        valid_groups = {g: a for g, a in group_aucs.items() if a is not None}

        if valid_groups:
            plt.figure(figsize=(8, 5))
            groups = list(valid_groups.keys())
            aucs = list(valid_groups.values())

            bars = plt.bar(groups, aucs, color='coral', edgecolor='black')
            plt.axhline(0.5, color='gray', linestyle='--', label='Random')
            plt.xlabel('Group')
            plt.ylabel('AUC')
            plt.title('AUC by Group (within-group)')
            plt.ylim(0, 1)

            for bar, auc in zip(bars, aucs):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{auc:.2f}', ha='center')

            plt.legend()
            plt.tight_layout()
            plt.savefig(viz_dir / 'auc_by_group.png', dpi=150)
            plt.close()

    logger.info(f"Saved visualizations to {viz_dir}")


def save_results(
    output_dir: str,
    cv_metrics: pd.DataFrame,
    metabolite_rank: pd.DataFrame,
    enzyme_rank: pd.DataFrame,
    group_metrics: Optional[pd.DataFrame] = None,
    permutation_results: Optional[pd.DataFrame] = None,
    sanity_results: Optional[pd.DataFrame] = None
):
    """
    Save all result files.

    Args:
        output_dir: Output directory
        cv_metrics: CV metrics DataFrame
        metabolite_rank: Metabolite ranking DataFrame
        enzyme_rank: Enzyme ranking DataFrame
        group_metrics: Per-group metrics DataFrame
        permutation_results: Permutation test results
        sanity_results: Sanity check results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cv_metrics.to_csv(output_path / 'cv_metrics_overall.csv', index=False)
    metabolite_rank.to_csv(output_path / 'metabolite_rank.csv', index=False)
    enzyme_rank.to_csv(output_path / 'enzyme_rank.csv', index=False)

    if group_metrics is not None:
        group_metrics.to_csv(output_path / 'cv_metrics_by_group.csv', index=False)

    if permutation_results is not None:
        permutation_results.to_csv(output_path / 'permutation_auc_distribution.csv', index=False)

    if sanity_results is not None:
        sanity_results.to_csv(output_path / 'sanity_check_results.csv', index=False)

    logger.info(f"Saved results to {output_path}")
