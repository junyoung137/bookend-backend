# package repositories
"""
Repositories package for Bookend Recommendation System.

Provides repository pattern implementations for all models.
"""

from .base_repository import (
    BaseRepository,
    RepositoryError,
    RecordNotFoundError,
    DuplicateRecordError
)
from .user_repository import UserRepository
from .item_repository import ItemRepository
from .interaction_repository import InteractionRepository

__all__ = [
    # Base
    "BaseRepository",
    "RepositoryError",
    "RecordNotFoundError",
    "DuplicateRecordError",
    
    # Concrete repositories
    "UserRepository",
    "ItemRepository",
    "InteractionRepository",
]

__version__ = "0.1.0"