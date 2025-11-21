# package loaders
"""
Data loaders package for Bookend Recommendation System.

This package provides loader implementations for:
- PostgreSQL bulk loading
- MinIO object storage (future)
- Cache warming (future)
"""

from .postgres_loader import PostgresLoader

__all__ = [
    "PostgresLoader",
]