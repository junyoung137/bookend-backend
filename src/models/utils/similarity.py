"""
Similarity computation utilities for recommendation systems.

Provides multiple similarity metrics:
1. Cosine Similarity - For dense vectors
2. Jaccard Similarity - For binary/set data
3. Pearson Correlation - For rating data

Principles:
- Single Responsibility - Each function computes one metric
- Error Handling - Graceful fallback for edge cases
- Performance - Vectorized operations where possible
- Type Safety - Clear input/output types
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from enum import Enum
import logging
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)


# =========================================================
# Similarity Metric Enum
# =========================================================

class SimilarityMetric(str, Enum):
    """Available similarity metrics."""
    COSINE = "cosine"
    JACCARD = "jaccard"
    PEARSON = "pearson"
    EUCLIDEAN = "euclidean"


# =========================================================
# Cosine Similarity
# =========================================================

def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray, handle_zero: bool = True) -> float:
    try:
        if vector_a.shape != vector_b.shape:
            raise ValueError(f"Vector dimensions mismatch: {vector_a.shape} vs {vector_b.shape}")

        norm_a, norm_b = np.linalg.norm(vector_a), np.linalg.norm(vector_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0 if handle_zero else ValueError("Zero vector encountered")

        similarity = 1.0 - cosine(vector_a, vector_b)
        return float(np.clip(similarity, -1.0, 1.0))
    except Exception as e:
        logger.error(f"Failed to compute cosine similarity: {e}")
        return 0.0


def cosine_similarity_matrix(vectors: np.ndarray, handle_zero: bool = True) -> np.ndarray:
    try:
        n = vectors.shape[0]
        sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                sim = 1.0 if i == j else cosine_similarity(vectors[i], vectors[j], handle_zero)
                sim_matrix[i, j] = sim_matrix[j, i] = sim
        return sim_matrix
    except Exception as e:
        logger.error(f"Failed to compute cosine similarity matrix: {e}")
        return np.zeros((vectors.shape[0], vectors.shape[0]))


# =========================================================
# Jaccard Similarity
# =========================================================

def jaccard_similarity(set_a: Set[Any], set_b: Set[Any], handle_empty: bool = True) -> float:
    try:
        if not set_a or not set_b:
            return 0.0 if handle_empty else ValueError("Empty set encountered")
        return float(len(set_a & set_b) / len(set_a | set_b))
    except Exception as e:
        logger.error(f"Failed to compute Jaccard similarity: {e}")
        return 0.0


def jaccard_similarity_from_binary(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    try:
        if vector_a.shape != vector_b.shape:
            raise ValueError("Vector dimensions mismatch")
        set_a = set(np.where(vector_a > 0)[0])
        set_b = set(np.where(vector_b > 0)[0])
        return jaccard_similarity(set_a, set_b)
    except Exception as e:
        logger.error(f"Failed to compute Jaccard similarity from binary: {e}")
        return 0.0


# =========================================================
# Pearson Correlation
# =========================================================

def pearson_correlation(vector_a: np.ndarray, vector_b: np.ndarray, min_overlap: int = 2, handle_constant: bool = True) -> float:
    try:
        if vector_a.shape != vector_b.shape:
            raise ValueError("Vector dimensions mismatch")

        mask = (vector_a != 0) & (vector_b != 0)
        if np.sum(mask) < min_overlap:
            return 0.0

        a, b = vector_a[mask], vector_b[mask]
        if np.std(a) == 0 or np.std(b) == 0:
            return 0.0 if handle_constant else ValueError("Constant vector encountered")

        corr, _p_value = pearsonr(a, b)
        return 0.0 if np.isnan(corr) else float(corr)
    except Exception as e:
        logger.error(f"Failed to compute Pearson correlation: {e}")
        return 0.0


# =========================================================
# Generic Similarity Computation
# =========================================================

def compute_similarity(vector_a: np.ndarray, vector_b: np.ndarray, metric: SimilarityMetric = SimilarityMetric.COSINE, **kwargs) -> float:
    try:
        if metric == SimilarityMetric.COSINE:
            return cosine_similarity(vector_a, vector_b, **kwargs)
        elif metric == SimilarityMetric.JACCARD:
            return jaccard_similarity_from_binary(vector_a, vector_b)
        elif metric == SimilarityMetric.PEARSON:
            return pearson_correlation(vector_a, vector_b, **kwargs)
        elif metric == SimilarityMetric.EUCLIDEAN:
            return float(np.exp(-np.linalg.norm(vector_a - vector_b)))
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")
    except Exception as e:
        logger.error(f"Failed to compute similarity: {e}")
        return 0.0


def compute_similarity_matrix(vectors: np.ndarray, metric: SimilarityMetric = SimilarityMetric.COSINE, **kwargs) -> np.ndarray:
    try:
        n = vectors.shape[0]
        sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                sim = 1.0 if i == j else compute_similarity(vectors[i], vectors[j], metric=metric, **kwargs)
                sim_matrix[i, j] = sim_matrix[j, i] = sim
        return sim_matrix
    except Exception as e:
        logger.error(f"Failed to compute similarity matrix: {e}")
        return np.zeros((vectors.shape[0], vectors.shape[0]))


# =========================================================
# Top-K Similarity Retrieval
# =========================================================

def get_top_k_similar(query_vector: np.ndarray, candidate_vectors: np.ndarray, k: int = 10, metric: SimilarityMetric = SimilarityMetric.COSINE, exclude_indices: Optional[List[int]] = None) -> List[Tuple[int, float]]:
    try:
        exclude_indices = exclude_indices or []
        sims = [(i, compute_similarity(query_vector, v, metric=metric))
                for i, v in enumerate(candidate_vectors) if i not in exclude_indices]
        return sorted(sims, key=lambda x: x[1], reverse=True)[:k]
    except Exception as e:
        logger.error(f"Failed to get top-k similar: {e}")
        return []
