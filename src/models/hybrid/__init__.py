# package hybrid
"""
Hybrid recommendation models package for Bookend Recommendation System.

This package provides:
- Hybrid recommender (User-CF + Item-CF + Context)
- Ambient recommender (layout-aware recommendations)
- Temporal recommender (time-aware recommendations)
- Context echo recommender (style-aware recommendations)
- Soft loop recommender (diversity-aware recommendations)
"""

from .hybrid_recommender import HybridRecommender
from .ambient_recommender import AmbientRecommender
from .temporal_recommender import TemporalRecommender

__all__ = [
    "HybridRecommender",
    "AmbientRecommender",
    "TemporalRecommender",
]