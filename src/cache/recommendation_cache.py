# src/cache/recommendation_cache.py
"""
Recommendation-specific caching layer.

Features:
1. Cache recommendation results by user and context
2. Cache user features and patterns
3. Cache similarity matrices
4. Smart invalidation on user activity
5. Cache warming strategies

Principles:
- Single Responsibility: Only handles recommendation caching
- Error Handling: Never fail requests due to cache errors
- Performance: Minimize cache key collisions
- Consistency: Automatic invalidation on data changes
"""

from typing import Optional, List, Dict, Any, Callable
from datetime import datetime
import logging
import hashlib
import json

from src.cache.redis_client import RedisClient, get_redis_client, CacheConfig
from src.models.base_recommender import RecommendationResult

logger = logging.getLogger(__name__)


# =========================================================
# Cache Key Builder
# =========================================================

class CacheKeyBuilder:
    """
    Build consistent cache keys for different data types.
    
    Key format: <namespace>:<entity>:<id>:<variant>
    Example: "bookend:user:123:features"
    """
    
    NAMESPACE = "bookend"
    
    @classmethod
    def user_features(cls, user_id: int) -> str:
        """Cache key for user features."""
        return f"{cls.NAMESPACE}:user:{user_id}:features"
    
    @classmethod
    def user_recommendations(
        cls,
        user_id: int,
        model_name: str,
        context_hash: Optional[str] = None
    ) -> str:
        """Cache key for user recommendations."""
        base = f"{cls.NAMESPACE}:user:{user_id}:recommendations:{model_name}"
        if context_hash:
            return f"{base}:{context_hash}"
        return base
    
    @classmethod
    def ambient_layout(
        cls,
        user_id: int,
        layout_type: str = "standard",
        slot_type: str = "hero_banner"
    ) -> str:
        """Cache key for ambient layout recommendations."""
        return (
            f"{cls.NAMESPACE}:user:{user_id}:ambient:"
            f"{layout_type}:{slot_type}"
        )
    
    @classmethod
    def temporal_pattern(cls, user_id: int) -> str:
        """Cache key for user temporal patterns."""
        return f"{cls.NAMESPACE}:user:{user_id}:temporal_pattern"
    
    @classmethod
    def item_features(cls, item_id: int) -> str:
        """Cache key for item features."""
        return f"{cls.NAMESPACE}:item:{item_id}:features"
    
    @classmethod
    def similarity_matrix(cls, matrix_type: str) -> str:
        """Cache key for similarity matrices."""
        return f"{cls.NAMESPACE}:matrix:{matrix_type}"
    
    @classmethod
    def user_session(cls, user_id: int, session_id: str) -> str:
        """Cache key for user session data."""
        return f"{cls.NAMESPACE}:session:{user_id}:{session_id}"
    
    @classmethod
    def model_prediction(
        cls,
        model_name: str,
        input_hash: str
    ) -> str:
        """Cache key for model predictions."""
        return f"{cls.NAMESPACE}:prediction:{model_name}:{input_hash}"
    
    @classmethod
    def compute_context_hash(cls, context: Optional[Dict[str, Any]]) -> str:
        """
        Compute deterministic hash of context dictionary.
        
        Args:
            context: Context dictionary
        
        Returns:
            Hash string (first 16 chars of SHA256)
        """
        if not context:
            return "default"
        
        try:
            # Sort keys for consistent hashing
            context_str = json.dumps(context, sort_keys=True)
            hash_obj = hashlib.sha256(context_str.encode())
            return hash_obj.hexdigest()[:16]
        except Exception as e:
            logger.error(f"Context hash failed: {e}")
            return "error"


# =========================================================
# Recommendation Cache
# =========================================================

class RecommendationCache:
    """
    High-level caching interface for recommendation system.
    
    Features:
    - Recommendation result caching
    - User feature caching
    - Automatic cache invalidation
    - Cache statistics tracking
    """
    
    def __init__(self, redis_client: Optional[RedisClient] = None):
        """
        Initialize recommendation cache.
        
        Args:
            redis_client: RedisClient instance (creates if None)
        """
        self.redis = redis_client or get_redis_client()
        self.key_builder = CacheKeyBuilder()
        
        # Statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "invalidations": 0,
        }
        
        logger.info("Recommendation cache initialized")
    
    # =========================================================
    # Recommendation Caching
    # =========================================================
    
    def get_recommendations(
        self,
        user_id: int,
        model_name: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[List[RecommendationResult]]:
        """
        Get cached recommendations for user.
        
        Args:
            user_id: User database ID
            model_name: Recommender model name
            context: Context dictionary
        
        Returns:
            List of RecommendationResult or None if not cached
        
        Example:
            >>> cache = get_recommendation_cache()
            >>> results = cache.get_recommendations(
            ...     user_id=123,
            ...     model_name="hybrid_recommender",
            ...     context={"time_of_day": "morning"}
            ... )
        """
        try:
            # Build cache key
            context_hash = self.key_builder.compute_context_hash(context)
            cache_key = self.key_builder.user_recommendations(
                user_id,
                model_name,
                context_hash
            )
            
            # Get from cache
            cached_data = self.redis.get(cache_key)
            
            if cached_data is None:
                self._stats["misses"] += 1
                logger.debug(f"Cache miss: {cache_key}")
                return None
            
            # Deserialize recommendations
            recommendations = self._deserialize_recommendations(cached_data)
            
            if recommendations:
                self._stats["hits"] += 1
                logger.debug(
                    f"Cache hit: {cache_key} "
                    f"({len(recommendations)} recommendations)"
                )
            
            return recommendations
        
        except Exception as e:
            logger.error(f"Get recommendations from cache failed: {e}", exc_info=True)
            return None
    
    def set_recommendations(
        self,
        user_id: int,
        model_name: str,
        recommendations: List[RecommendationResult],
        context: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache recommendations for user.
        
        Args:
            user_id: User database ID
            model_name: Recommender model name
            recommendations: List of RecommendationResult
            context: Context dictionary
            ttl: Time-to-live in seconds (None = use default)
        
        Returns:
            True if cached successfully
        
        Example:
            >>> cache = get_recommendation_cache()
            >>> cache.set_recommendations(
            ...     user_id=123,
            ...     model_name="hybrid_recommender",
            ...     recommendations=[...],
            ...     ttl=1800
            ... )
        """
        try:
            # Build cache key
            context_hash = self.key_builder.compute_context_hash(context)
            cache_key = self.key_builder.user_recommendations(
                user_id,
                model_name,
                context_hash
            )
            
            # Serialize recommendations
            serialized = self._serialize_recommendations(recommendations)
            
            # Set TTL
            if ttl is None:
                ttl = CacheConfig.RECOMMENDATION_TTL
            
            # Cache
            success = self.redis.set(cache_key, serialized, ttl=ttl)
            
            if success:
                self._stats["sets"] += 1
                logger.debug(
                    f"Cached {len(recommendations)} recommendations "
                    f"for user {user_id} (ttl={ttl}s)"
                )
            
            return success
        
        except Exception as e:
            logger.error(f"Set recommendations to cache failed: {e}", exc_info=True)
            return False
    
    def invalidate_user_recommendations(
        self,
        user_id: int,
        model_name: Optional[str] = None
    ) -> int:
        """
        Invalidate cached recommendations for user.
        
        Args:
            user_id: User database ID
            model_name: Specific model to invalidate (None = all models)
        
        Returns:
            Number of keys invalidated
        
        Example:
            >>> cache = get_recommendation_cache()
            >>> # Invalidate all recommendations for user
            >>> cache.invalidate_user_recommendations(user_id=123)
            >>> 
            >>> # Invalidate only hybrid recommendations
            >>> cache.invalidate_user_recommendations(
            ...     user_id=123,
            ...     model_name="hybrid_recommender"
            ... )
        """
        try:
            if model_name:
                # Invalidate specific model
                pattern = f"{self.key_builder.NAMESPACE}:user:{user_id}:recommendations:{model_name}:*"
            else:
                # Invalidate all models
                pattern = f"{self.key_builder.NAMESPACE}:user:{user_id}:recommendations:*"
            
            count = self.redis.delete_pattern(pattern)
            
            if count > 0:
                self._stats["invalidations"] += count
                logger.info(
                    f"Invalidated {count} recommendation caches for user {user_id}"
                )
            
            return count
        
        except Exception as e:
            logger.error(f"Invalidate user recommendations failed: {e}", exc_info=True)
            return 0
    
    # =========================================================
    # Ambient Layout Caching
    # =========================================================
    
    def get_ambient_layout(
        self,
        user_id: int,
        layout_type: str = "standard",
        slot_type: str = "hero_banner"
    ) -> Optional[List[int]]:
        """
        Get cached ambient layout (item IDs).
        
        Args:
            user_id: User database ID
            layout_type: Layout type
            slot_type: Slot type
        
        Returns:
            List of item IDs or None if not cached
        """
        try:
            cache_key = self.key_builder.ambient_layout(
                user_id,
                layout_type,
                slot_type
            )
            
            cached_data = self.redis.get(cache_key)
            
            if cached_data is None:
                self._stats["misses"] += 1
                return None
            
            self._stats["hits"] += 1
            return cached_data  # Should be list of ints
        
        except Exception as e:
            logger.error(f"Get ambient layout from cache failed: {e}", exc_info=True)
            return None
    
    def set_ambient_layout(
        self,
        user_id: int,
        item_ids: List[int],
        layout_type: str = "standard",
        slot_type: str = "hero_banner",
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache ambient layout.
        
        Args:
            user_id: User database ID
            item_ids: List of item IDs for layout
            layout_type: Layout type
            slot_type: Slot type
            ttl: Time-to-live in seconds
        
        Returns:
            True if cached successfully
        """
        try:
            cache_key = self.key_builder.ambient_layout(
                user_id,
                layout_type,
                slot_type
            )
            
            if ttl is None:
                ttl = CacheConfig.AMBIENT_LAYOUT_TTL
            
            success = self.redis.set(cache_key, item_ids, ttl=ttl)
            
            if success:
                self._stats["sets"] += 1
                logger.debug(
                    f"Cached ambient layout for user {user_id} "
                    f"({len(item_ids)} items, ttl={ttl}s)"
                )
            
            return success
        
        except Exception as e:
            logger.error(f"Set ambient layout to cache failed: {e}", exc_info=True)
            return False
    
    # =========================================================
    # User Features Caching
    # =========================================================
    
    def get_user_features(self, user_id: int) -> Optional[Dict[str, Any]]:
        """
        Get cached user features.
        
        Args:
            user_id: User database ID
        
        Returns:
            User features dictionary or None
        """
        try:
            cache_key = self.key_builder.user_features(user_id)
            cached_data = self.redis.get(cache_key)
            
            if cached_data is None:
                self._stats["misses"] += 1
                return None
            
            self._stats["hits"] += 1
            return cached_data
        
        except Exception as e:
            logger.error(f"Get user features from cache failed: {e}", exc_info=True)
            return None
    
    def set_user_features(
        self,
        user_id: int,
        features: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache user features.
        
        Args:
            user_id: User database ID
            features: User features dictionary
            ttl: Time-to-live in seconds
        
        Returns:
            True if cached successfully
        """
        try:
            cache_key = self.key_builder.user_features(user_id)
            
            if ttl is None:
                ttl = CacheConfig.USER_FEATURES_TTL
            
            success = self.redis.set(cache_key, features, ttl=ttl)
            
            if success:
                self._stats["sets"] += 1
            
            return success
        
        except Exception as e:
            logger.error(f"Set user features to cache failed: {e}", exc_info=True)
            return False
    
    # =========================================================
    # Temporal Pattern Caching
    # =========================================================
    
    def get_temporal_pattern(self, user_id: int) -> Optional[Dict[str, Any]]:
        """
        Get cached temporal pattern for user.
        
        Args:
            user_id: User database ID
        
        Returns:
            Temporal pattern dictionary or None
        """
        try:
            cache_key = self.key_builder.temporal_pattern(user_id)
            cached_data = self.redis.get(cache_key)
            
            if cached_data is None:
                self._stats["misses"] += 1
                return None
            
            self._stats["hits"] += 1
            return cached_data
        
        except Exception as e:
            logger.error(f"Get temporal pattern from cache failed: {e}", exc_info=True)
            return None
    
    def set_temporal_pattern(
        self,
        user_id: int,
        pattern: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache temporal pattern for user.
        
        Args:
            user_id: User database ID
            pattern: Temporal pattern dictionary
            ttl: Time-to-live in seconds
        
        Returns:
            True if cached successfully
        """
        try:
            cache_key = self.key_builder.temporal_pattern(user_id)
            
            if ttl is None:
                ttl = CacheConfig.TEMPORAL_PATTERN_TTL
            
            success = self.redis.set(cache_key, pattern, ttl=ttl)
            
            if success:
                self._stats["sets"] += 1
            
            return success
        
        except Exception as e:
            logger.error(f"Set temporal pattern to cache failed: {e}", exc_info=True)
            return False
    
    # =========================================================
    # User Session Caching
    # =========================================================
    
    def get_user_session(
        self,
        user_id: int,
        session_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached user session data.
        
        Args:
            user_id: User database ID
            session_id: Session identifier
        
        Returns:
            Session data dictionary or None
        """
        try:
            cache_key = self.key_builder.user_session(user_id, session_id)
            cached_data = self.redis.get(cache_key)
            
            if cached_data is None:
                self._stats["misses"] += 1
                return None
            
            self._stats["hits"] += 1
            return cached_data
        
        except Exception as e:
            logger.error(f"Get user session from cache failed: {e}", exc_info=True)
            return None
    
    def set_user_session(
        self,
        user_id: int,
        session_id: str,
        session_data: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache user session data.
        
        Args:
            user_id: User database ID
            session_id: Session identifier
            session_data: Session data dictionary
            ttl: Time-to-live in seconds
        
        Returns:
            True if cached successfully
        """
        try:
            cache_key = self.key_builder.user_session(user_id, session_id)
            
            if ttl is None:
                ttl = CacheConfig.USER_SESSION_TTL
            
            success = self.redis.set(cache_key, session_data, ttl=ttl)
            
            if success:
                self._stats["sets"] += 1
            
            return success
        
        except Exception as e:
            logger.error(f"Set user session to cache failed: {e}", exc_info=True)
            return False
    
    # =========================================================
    # Bulk Invalidation
    # =========================================================
    
    def invalidate_user_all(self, user_id: int) -> int:
        """
        Invalidate all caches for a user.
        
        Args:
            user_id: User database ID
        
        Returns:
            Number of keys invalidated
        """
        try:
            pattern = f"{self.key_builder.NAMESPACE}:user:{user_id}:*"
            count = self.redis.delete_pattern(pattern)
            
            if count > 0:
                self._stats["invalidations"] += count
                logger.info(f"Invalidated all caches for user {user_id} ({count} keys)")
            
            return count
        
        except Exception as e:
            logger.error(f"Invalidate user all failed: {e}", exc_info=True)
            return 0
    
    # =========================================================
    # Statistics
    # =========================================================
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Statistics dictionary
        """
        total_requests = self._stats["hits"] + self._stats["misses"]
        hit_rate = (
            self._stats["hits"] / total_requests
            if total_requests > 0
            else 0.0
        )
        
        return {
            **self._stats,
            "total_requests": total_requests,
            "hit_rate": round(hit_rate, 4),
        }
    
    def reset_stats(self) -> None:
        """Reset cache statistics."""
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "invalidations": 0,
        }
        logger.info("Cache statistics reset")
    
    # =========================================================
    # Serialization Helpers
    # =========================================================
    
    def _serialize_recommendations(
        self,
        recommendations: List[RecommendationResult]
    ) -> List[Dict[str, Any]]:
        """
        Serialize recommendations for caching.
        
        Args:
            recommendations: List of RecommendationResult
        
        Returns:
            List of dictionaries
        """
        try:
            return [rec.to_dict() for rec in recommendations]
        except Exception as e:
            logger.error(f"Serialize recommendations failed: {e}")
            return []
    
    def _deserialize_recommendations(
        self,
        data: List[Dict[str, Any]]
    ) -> Optional[List[RecommendationResult]]:
        """
        Deserialize recommendations from cache.
        
        Args:
            data: List of dictionaries
        
        Returns:
            List of RecommendationResult or None
        """
        try:
            if not isinstance(data, list):
                return None
            
            recommendations = []
            
            for item_dict in data:
                # Convert back to RecommendationResult
                rec = RecommendationResult(
                    item_id=item_dict["item_id"],
                    item_code=item_dict["item_code"],
                    item_name=item_dict["item_name"],
                    score=item_dict["score"],
                    rank=item_dict["rank"],
                    reason=item_dict["reason"],
                    metadata=item_dict.get("metadata", {}),
                    timestamp=datetime.fromisoformat(item_dict["timestamp"])
                )
                recommendations.append(rec)
            
            return recommendations
        
        except Exception as e:
            logger.error(f"Deserialize recommendations failed: {e}", exc_info=True)
            return None


# =========================================================
# Singleton Getter
# =========================================================

_recommendation_cache_instance: Optional[RecommendationCache] = None


def get_recommendation_cache() -> RecommendationCache:
    """
    Get singleton recommendation cache instance.
    
    Returns:
        RecommendationCache instance
    
    Example:
        >>> cache = get_recommendation_cache()
        >>> results = cache.get_recommendations(
        ...     user_id=123,
        ...     model_name="hybrid_recommender"
        ... )
    """
    global _recommendation_cache_instance
    
    if _recommendation_cache_instance is None:
        _recommendation_cache_instance = RecommendationCache()
    
    return _recommendation_cache_instance


# =========================================================
# Testing Utilities
# =========================================================

if __name__ == "__main__":
    from config.logging_config import setup_logging
    
    setup_logging()
    
    print("\n" + "=" * 70)
    print("💾 RECOMMENDATION CACHE TEST")
    print("=" * 70)
    
    # Get cache instance
    cache = get_recommendation_cache()
    
    # Test cache key building
    print("\n🔑 Testing cache key building...")
    key_builder = CacheKeyBuilder()
    
    keys = {
        "user_features": key_builder.user_features(123),
        "recommendations": key_builder.user_recommendations(
            123,
            "hybrid_recommender",
            "abc123"
        ),
        "ambient_layout": key_builder.ambient_layout(123),
        "temporal_pattern": key_builder.temporal_pattern(123),
    }
    
    for name, key in keys.items():
        print(f"  {name}: {key}")
    
    # Test context hashing
    print("\n#️⃣ Testing context hashing...")
    context1 = {"time_of_day": "morning", "device": "mobile"}
    context2 = {"device": "mobile", "time_of_day": "morning"}
    
    hash1 = key_builder.compute_context_hash(context1)
    hash2 = key_builder.compute_context_hash(context2)
    
    print(f"  Context 1: {hash1}")
    print(f"  Context 2: {hash2}")
    print(f"  Hashes match: {'✅' if hash1 == hash2 else '❌'}")
    
    # Test user features caching
    print("\n👤 Testing user features caching...")
    
    test_features = {
        "total_paraphrases": 100,
        "preferred_tone": "formal",
        "last_7d_count": 25
    }
    
    cache.set_user_features(999, test_features, ttl=60)
    print(f"  Set features: ✅")
    
    cached_features = cache.get_user_features(999)
    print(f"  Get features: {cached_features}")
    
    # Test statistics
    print("\n📊 Cache statistics:")
    stats = cache.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Cleanup
    cache.invalidate_user_all(999)
    print("\n🧹 Cleanup: ✅")
    
    print("\n" + "=" * 70)
    print("✅ Recommendation cache test complete")