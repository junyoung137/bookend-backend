from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
import logging
import numpy as np

logger = logging.getLogger(__name__)

# =========================================================
# Normalization Methods Enum
# =========================================================

class NormalizationMethod(str, Enum):
    MIN_MAX = "min_max"
    Z_SCORE = "z_score"
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"
    RANK = "rank"
    NONE = "none"


# =========================================================
# Core Normalization Functions
# =========================================================

def normalize_scores(
    scores: Dict[int, float],
    method: Union[str, NormalizationMethod] = NormalizationMethod.MIN_MAX,
    clip_range: Optional[Tuple[float, float]] = None
) -> Dict[int, float]:
    if not scores:
        logger.warning("Empty scores dictionary provided")
        return {}

    try:
        # Convert to enum
        if isinstance(method, str):
            method = NormalizationMethod(method)

        item_ids = list(scores.keys())
        values = np.asarray(list(scores.values()), dtype=float)

        # Normalization switch
        if method == NormalizationMethod.MIN_MAX:
            normalized = _min_max_normalize(values)
        elif method == NormalizationMethod.Z_SCORE:
            normalized = _z_score_normalize(values)
        elif method == NormalizationMethod.SIGMOID:
            normalized = _sigmoid_normalize(values)
        elif method == NormalizationMethod.SOFTMAX:
            normalized = _softmax_normalize(values)
        elif method == NormalizationMethod.RANK:
            normalized = _rank_normalize(values)
        elif method == NormalizationMethod.NONE:
            normalized = values
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        if clip_range:
            normalized = np.clip(normalized, *clip_range)

        normalized_scores = {k: float(v) for k, v in zip(item_ids, normalized)}
        logger.debug(f"Normalized {len(scores)} scores with {method.value}")
        return normalized_scores

    except Exception as e:
        logger.exception(f"Normalization failed: {e}")
        return scores


def _min_max_normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    min_val, max_val = np.min(values), np.max(values)
    if max_val == min_val:
        return np.full_like(values, 0.5)
    return (values - min_val) / (max_val - min_val)


def _z_score_normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    mean, std = np.mean(values), np.std(values)
    if std == 0:
        return np.zeros_like(values)
    return (values - mean) / std


def _sigmoid_normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    z = _z_score_normalize(values)
    return 1.0 / (1.0 + np.exp(-z))


def _softmax_normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    exps = np.exp(values - np.max(values))
    return exps / np.sum(exps)


def _rank_normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    ranks = np.argsort(np.argsort(values)) + 1
    return ranks / len(values)


# =========================================================
# Score Blending Functions
# =========================================================

def blend_scores(
    score_dicts: List[Dict[int, float]],
    weights: Optional[List[float]] = None,
    method: str = "weighted_average",
    normalize_before_blend: bool = True
) -> Dict[int, float]:
    if not score_dicts:
        logger.warning("Empty score_dicts list provided")
        return {}

    try:
        if weights is None:
            weights = [1.0 / len(score_dicts)] * len(score_dicts)

        if len(weights) != len(score_dicts):
            raise ValueError("Weights length must match number of score dicts")

        total_weight = sum(weights)
        if not np.isclose(total_weight, 1.0):
            logger.warning(f"Weights sum={total_weight:.3f}, normalizing to 1.0")
            weights = [w / total_weight for w in weights]

        if normalize_before_blend:
            score_dicts = [normalize_scores(sd, "min_max") for sd in score_dicts]

        if method == "weighted_average":
            return _weighted_average_blend(score_dicts, weights)
        elif method == "rank_fusion":
            return _rank_fusion_blend(score_dicts, weights)
        elif method == "borda":
            return _borda_blend(score_dicts, weights)
        else:
            raise ValueError(f"Unknown blending method: {method}")

    except Exception as e:
        logger.exception(f"Score blending failed: {e}")
        return next((sd for sd in score_dicts if sd), {})


def _weighted_average_blend(score_dicts: List[Dict[int, float]], weights: List[float]) -> Dict[int, float]:
    blended = {}
    all_items = set().union(*score_dicts)
    for item in all_items:
        vals, w_sum = [], []
        for sd, w in zip(score_dicts, weights):
            if item in sd:
                vals.append(sd[item] * w)
                w_sum.append(w)
        if w_sum:
            blended[item] = sum(vals) / sum(w_sum)
    return blended


def _rank_fusion_blend(score_dicts: List[Dict[int, float]], weights: List[float]) -> Dict[int, float]:
    k = 60
    blended = {}
    all_items = set().union(*score_dicts)
    for item in all_items:
        score = 0.0
        for sd, w in zip(score_dicts, weights):
            if item in sd:
                ranked = sorted(sd.items(), key=lambda x: x[1], reverse=True)
                rank = next((i + 1 for i, (iid, _) in enumerate(ranked) if iid == item), len(ranked) + 1)
                score += w / (k + rank)
        blended[item] = score
    return blended


def _borda_blend(score_dicts: List[Dict[int, float]], weights: List[float]) -> Dict[int, float]:
    blended = {}
    all_items = set().union(*score_dicts)
    for item in all_items:
        total = 0.0
        for sd, w in zip(score_dicts, weights):
            if item in sd:
                ranked = sorted(sd.items(), key=lambda x: x[1], reverse=True)
                rank = next((i for i, (iid, _) in enumerate(ranked) if iid == item), len(ranked))
                total += w * (len(sd) - rank)
        blended[item] = total
    return blended


# =========================================================
# Score Transformation Functions
# =========================================================

def transform_scores(
    scores: Dict[int, float],
    transformation: str,
    **kwargs
) -> Dict[int, float]:
    if not scores:
        return {}

    try:
        ids = list(scores.keys())
        vals = np.asarray(list(scores.values()), dtype=float)

        if transformation == "exp":
            scale = kwargs.get("scale", 1.0)
            out = np.exp(scale * vals)
        elif transformation == "log":
            scale = kwargs.get("scale", 1.0)
            out = np.log1p(scale * vals)
        elif transformation == "power":
            exp = kwargs.get("exponent", 2.0)
            out = np.power(vals, exp)
        elif transformation == "threshold":
            t = kwargs.get("threshold", 0.5)
            out = np.where(vals >= t, vals, 0.0)
        else:
            raise ValueError(f"Unknown transformation: {transformation}")

        return {i: float(v) for i, v in zip(ids, out)}

    except Exception as e:
        logger.exception(f"Score transformation failed: {e}")
        return scores


# =========================================================
# Score Filtering & Statistics
# =========================================================

def filter_scores(
    scores: Dict[int, float],
    min_score: Optional[float] = None,
    max_score: Optional[float] = None,
    top_k: Optional[int] = None,
    min_count: Optional[int] = None
) -> Dict[int, float]:
    if not scores:
        return {}

    try:
        filtered = {k: v for k, v in scores.items()
                    if (min_score is None or v >= min_score)
                    and (max_score is None or v <= max_score)}

        if top_k is not None:
            filtered = dict(sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:top_k])

        if min_count is not None and len(filtered) < min_count:
            sorted_all = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            filtered = dict(sorted_all[:min_count])

        logger.debug(f"Filtered scores: {len(scores)} → {len(filtered)}")
        return filtered

    except Exception as e:
        logger.exception(f"Score filtering failed: {e}")
        return scores


def compute_score_statistics(scores: Dict[int, float]) -> Dict[str, float]:
    if not scores:
        return dict(count=0, mean=0, std=0, min=0, max=0, median=0, q25=0, q75=0)

    vals = np.asarray(list(scores.values()), dtype=float)
    return {
        "count": len(vals),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "median": float(np.median(vals)),
        "q25": float(np.percentile(vals, 25)),
        "q75": float(np.percentile(vals, 75)),
    }


if __name__ == "__main__":
    print("=" * 70)
    print("SCORING UTILITIES DEMO")
    print("=" * 70)

    scores = {1: 10, 2: 20, 3: 30, 4: 40, 5: 50}
    print("Min-Max:", normalize_scores(scores, "min_max"))
    print("Z-Score:", normalize_scores(scores, "z_score"))
    print("Softmax:", normalize_scores(scores, "softmax"))

    s1 = {1: 0.9, 2: 0.7, 3: 0.5}
    s2 = {1: 0.8, 2: 0.9, 3: 0.6, 4: 0.7}
    print("Blend:", blend_scores([s1, s2], weights=[0.6, 0.4]))

    print("Transform:", transform_scores({1: 0.5, 2: 0.7}, "exp", scale=2))
    print("Filter:", filter_scores({1: 0.9, 2: 0.7, 3: 0.3}, min_score=0.5, top_k=2))
    print("Stats:", compute_score_statistics({1: 0.9, 2: 0.7, 3: 0.5}))
