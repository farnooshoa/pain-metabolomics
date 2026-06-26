"""
Pain Heterogeneous GNN Pipeline

A complete pipeline for pain classification using metabolomics data
with LLM-based name normalization and KEGG REST API mapping.
"""

__version__ = "1.0.0"
__author__ = "Pain GNN Pipeline"

from .openrouter_aliases import OpenRouterAliasGenerator
from .kegg_mapper import KEGGMapper
from .utils import (
    load_metabolomics_data,
    preprocess_fold,
    build_sample_features,
    compute_group_confounding,
    compute_within_group_auc,
    create_visualizations,
    save_results
)

__all__ = [
    "OpenRouterAliasGenerator",
    "KEGGMapper",
    "load_metabolomics_data",
    "preprocess_fold",
    "build_sample_features",
    "compute_group_confounding",
    "compute_within_group_auc",
    "create_visualizations",
    "save_results"
]
