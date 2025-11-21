# package utils
"""
Model utilities package for Bookend Recommendation System.

This package provides:
- Similarity computation (cosine, jaccard, pearson)
- MMR (Maximal Marginal Relevance) for diversity
- Scoring utilities (normalization, blending)
- Evaluation metrics (precision, recall, NDCG)
"""

from .similarity import (
    cosine_similarity,
    jaccard_similarity,
    pearson_correlation,
    compute_similarity_matrix,
    SimilarityMetric
)

__all__ = [
    "cosine_similarity",
    "jaccard_similarity",
    "pearson_correlation",
    "compute_similarity_matrix",
    "SimilarityMetric",
]