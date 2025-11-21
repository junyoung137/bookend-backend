# src/cache/__init__.py
"""
Cache module for Bookend Recommendation System.

Provides:
- Redis client singleton
- Recommendation result caching
- User session caching
- Model prediction caching
- Cache invalidation strategies
"""

from .redis_client import (
    RedisClient,
    get_redis_client,
    CacheConfig,
)
from .recommendation_cache import (
    RecommendationCache,
    get_recommendation_cache,
)
from .decorators import (
    cache_result,
    invalidate_cache,
)

__all__ = [
    "RedisClient",
    "get_redis_client",
    "CacheConfig",
    "RecommendationCache",
    "get_recommendation_cache",
    "cache_result",
    "invalidate_cache",
]