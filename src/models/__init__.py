# models package
"""
Recommendation models package for Bookend Recommendation System.

This package provides:
- Base recommender abstract class
- Collaborative filtering (User-CF, Item-CF)
- Hybrid recommenders (Ambient, Temporal, Context Echo, etc.)
- Utility functions (similarity, scoring, MMR)
- Model evaluation tools
"""

from .base_recommender import BaseRecommender, RecommendationResult

__all__ = [
    "BaseRecommender",
    "RecommendationResult",
]
