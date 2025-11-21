# package collaborative
"""
Collaborative filtering models package for Bookend Recommendation System.

This package provides:
- User-based Collaborative Filtering (User-CF)
- Item-based Collaborative Filtering (Item-CF)
- Interaction matrix utilities
"""

from .user_cf import UserCFRecommender
from .item_cf import ItemCFRecommender
from .interaction_matrix import InteractionMatrixBuilder

__all__ = [
    "UserCFRecommender",
    "ItemCFRecommender",
    "InteractionMatrixBuilder",
]