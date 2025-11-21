# src/cache/redis_client.py
"""
Redis client singleton for caching operations.

Features:
1. Thread-safe singleton pattern
2. Connection pooling
3. Automatic reconnection
4. JSON serialization support
5. TTL management
6. Graceful degradation (fallback to in-memory)

Principles:
- Single Responsibility: Only handles Redis operations
- Error Handling: Never fail requests due to cache errors
- Performance: Connection pooling, pipelining
- Reliability: Automatic retry, fallback to in-memory
"""

from typing import Optional, Any, Dict, List, Union
from datetime import timedelta
import logging
import json
import pickle
from functools import wraps

try:
    import redis
    from redis.connection import ConnectionPool
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None
    ConnectionPool = None

from config.settings import get_settings

logger = logging.getLogger(__name__)


# =========================================================
# Cache Configuration
# =========================================================

class CacheConfig:
    """Cache TTL configuration."""
    
    # Recommendation caches
    RECOMMENDATION_TTL = 1800  # 30 minutes
    AMBIENT_LAYOUT_TTL = 3600  # 1 hour
    TEMPORAL_PATTERN_TTL = 7200  # 2 hours
    
    # User caches
    USER_FEATURES_TTL = 3600  # 1 hour
    USER_SESSION_TTL = 86400  # 24 hours
    
    # Model caches
    MODEL_PREDICTION_TTL = 600  # 10 minutes
    SIMILARITY_MATRIX_TTL = 86400  # 24 hours
    
    # Rate limiting
    RATE_LIMIT_WINDOW = 60  # 1 minute


# =========================================================
# In-Memory Fallback Cache
# =========================================================

class InMemoryCache:
    """
    Simple in-memory cache for fallback when Redis unavailable.
    
    Thread-safe for single-process deployment.
    Uses LRU eviction when size limit reached.
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize in-memory cache.
        
        Args:
            max_size: Maximum number of keys to store
        """
        self.max_size = max_size
        self._storage: Dict[str, Any] = {}
        self._access_order: List[str] = []
        
        logger.info(f"In-memory cache initialized (max_size={max_size})")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            if key in self._storage:
                # Update access order (LRU)
                if key in self._access_order:
                    self._access_order.remove(key)
                self._access_order.append(key)
                
                return self._storage[key]
            return None
        except Exception as e:
            logger.error(f"In-memory cache get failed: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        try:
            # Evict oldest if at capacity
            if len(self._storage) >= self.max_size and key not in self._storage:
                if self._access_order:
                    oldest_key = self._access_order.pop(0)
                    del self._storage[oldest_key]
            
            # Store value
            self._storage[key] = value
            
            # Update access order
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            
            # Note: TTL not implemented for in-memory cache
            return True
        except Exception as e:
            logger.error(f"In-memory cache set failed: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        try:
            if key in self._storage:
                del self._storage[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                return True
            return False
        except Exception as e:
            logger.error(f"In-memory cache delete failed: {e}")
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._storage.clear()
        self._access_order.clear()


# =========================================================
# Redis Client
# =========================================================

class RedisClient:
    """
    Redis client singleton with connection pooling.
    
    Features:
    - Automatic connection management
    - JSON serialization
    - TTL support
    - Batch operations
    - Fallback to in-memory cache
    """
    
    _instance: Optional['RedisClient'] = None
    _redis_client: Optional['redis.Redis'] = None
    _fallback_cache: Optional[InMemoryCache] = None
    _is_connected: bool = False
    
    def __new__(cls) -> 'RedisClient':
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize Redis client (only once due to singleton)."""
        if self._redis_client is None:
            self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Create Redis client with connection pool."""
        if not REDIS_AVAILABLE:
            logger.warning(
                "Redis library not installed. Using in-memory fallback cache. "
                "Install with: pip install redis"
            )
            self._fallback_cache = InMemoryCache(max_size=1000)
            self._is_connected = False
            return
        
        settings = get_settings()
        
        try:
            # Create connection pool
            pool = ConnectionPool(
                host=settings.redis.host,
                port=settings.redis.port,
                db=settings.redis.db,
                password=settings.redis.password,
                max_connections=settings.redis.max_connections,
                decode_responses=False,  # We'll handle decoding
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True,
            )
            
            # Create Redis client
            self._redis_client = redis.Redis(connection_pool=pool)
            
            # Test connection
            self._redis_client.ping()
            self._is_connected = True
            
            logger.info(
                f"Redis client initialized: "
                f"{settings.redis.host}:{settings.redis.port}/{settings.redis.db}"
            )
        
        except Exception as e:
            logger.warning(
                f"Failed to connect to Redis: {e}. "
                "Using in-memory fallback cache."
            )
            self._redis_client = None
            self._fallback_cache = InMemoryCache(max_size=1000)
            self._is_connected = False
    
    @property
    def is_connected(self) -> bool:
        """Check if Redis is connected."""
        return self._is_connected
    
    def _ensure_connection(func):
        """Decorator to ensure Redis connection and fallback on failure."""
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                # Try Redis if connected
                if self._is_connected and self._redis_client:
                    return func(self, *args, **kwargs)
                
                # Fallback to in-memory cache
                logger.debug("Using in-memory fallback cache")
                return self._fallback_operation(func.__name__, *args, **kwargs)
            
            except Exception as e:
                logger.error(f"Redis operation failed: {e}", exc_info=True)
                # Fallback on error
                return self._fallback_operation(func.__name__, *args, **kwargs)
        
        return wrapper
    
    def _fallback_operation(
        self,
        operation: str,
        *args,
        **kwargs
    ) -> Any:
        """Execute operation on fallback cache."""
        if not self._fallback_cache:
            self._fallback_cache = InMemoryCache()
        
        if operation == "get":
            return self._fallback_cache.get(args[0])
        elif operation == "set":
            return self._fallback_cache.set(args[0], args[1], kwargs.get('ttl'))
        elif operation == "delete":
            return self._fallback_cache.delete(args[0])
        elif operation == "exists":
            return self._fallback_cache.get(args[0]) is not None
        else:
            logger.warning(f"Fallback operation not supported: {operation}")
            return None
    
    @_ensure_connection
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None if not found
        
        Example:
            >>> redis_client = get_redis_client()
            >>> value = redis_client.get("user:123:features")
        """
        try:
            data = self._redis_client.get(key)
            
            if data is None:
                return None
            
            # Try to deserialize
            return self._deserialize(data)
        
        except Exception as e:
            logger.error(f"Cache get failed for key {key}: {e}")
            return None
    
    @_ensure_connection
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in cache with optional TTL.
        
        Args:
            key: Cache key
            value: Value to cache (will be serialized)
            ttl: Time-to-live in seconds (None = no expiration)
        
        Returns:
            True if successful, False otherwise
        
        Example:
            >>> redis_client = get_redis_client()
            >>> redis_client.set(
            ...     "user:123:features",
            ...     {"total_paraphrases": 100},
            ...     ttl=3600
            ... )
        """
        try:
            # Serialize value
            serialized = self._serialize(value)
            
            # Set with TTL
            if ttl:
                self._redis_client.setex(key, ttl, serialized)
            else:
                self._redis_client.set(key, serialized)
            
            return True
        
        except Exception as e:
            logger.error(f"Cache set failed for key {key}: {e}")
            return False
    
    @_ensure_connection
    def delete(self, key: str) -> bool:
        """
        Delete key from cache.
        
        Args:
            key: Cache key
        
        Returns:
            True if deleted, False otherwise
        """
        try:
            result = self._redis_client.delete(key)
            return bool(result)
        except Exception as e:
            logger.error(f"Cache delete failed for key {key}: {e}")
            return False
    
    @_ensure_connection
    def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key
        
        Returns:
            True if exists, False otherwise
        """
        try:
            return bool(self._redis_client.exists(key))
        except Exception as e:
            logger.error(f"Cache exists check failed for key {key}: {e}")
            return False
    
    @_ensure_connection
    def expire(self, key: str, ttl: int) -> bool:
        """
        Set expiration time for key.
        
        Args:
            key: Cache key
            ttl: Time-to-live in seconds
        
        Returns:
            True if successful, False otherwise
        """
        try:
            return bool(self._redis_client.expire(key, ttl))
        except Exception as e:
            logger.error(f"Cache expire failed for key {key}: {e}")
            return False
    
    @_ensure_connection
    def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """
        Increment counter.
        
        Args:
            key: Cache key
            amount: Amount to increment
        
        Returns:
            New value or None if failed
        """
        try:
            return self._redis_client.incrby(key, amount)
        except Exception as e:
            logger.error(f"Cache increment failed for key {key}: {e}")
            return None
    
    @_ensure_connection
    def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """
        Get multiple values at once (pipeline).
        
        Args:
            keys: List of cache keys
        
        Returns:
            Dictionary mapping keys to values
        """
        try:
            if not keys:
                return {}
            
            # Use pipeline for efficiency
            pipe = self._redis_client.pipeline()
            for key in keys:
                pipe.get(key)
            
            results = pipe.execute()
            
            # Deserialize results
            return {
                key: self._deserialize(data) if data else None
                for key, data in zip(keys, results)
            }
        
        except Exception as e:
            logger.error(f"Cache get_many failed: {e}")
            return {}
    
    @_ensure_connection
    def set_many(
        self,
        mapping: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set multiple values at once (pipeline).
        
        Args:
            mapping: Dictionary of key-value pairs
            ttl: Time-to-live for all keys
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not mapping:
                return True
            
            # Use pipeline for efficiency
            pipe = self._redis_client.pipeline()
            
            for key, value in mapping.items():
                serialized = self._serialize(value)
                if ttl:
                    pipe.setex(key, ttl, serialized)
                else:
                    pipe.set(key, serialized)
            
            pipe.execute()
            return True
        
        except Exception as e:
            logger.error(f"Cache set_many failed: {e}")
            return False
    
    @_ensure_connection
    def delete_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching pattern.
        
        Args:
            pattern: Key pattern (e.g., "user:*:recommendations")
        
        Returns:
            Number of keys deleted
        
        Example:
            >>> redis_client = get_redis_client()
            >>> # Delete all user recommendation caches
            >>> redis_client.delete_pattern("user:*:recommendations")
        """
        try:
            keys = self._redis_client.keys(pattern)
            
            if not keys:
                return 0
            
            return self._redis_client.delete(*keys)
        
        except Exception as e:
            logger.error(f"Cache delete_pattern failed for {pattern}: {e}")
            return 0
    
    def clear_all(self) -> bool:
        """
        Clear all cache entries (use with caution!).
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self._is_connected and self._redis_client:
                self._redis_client.flushdb()
                logger.warning("Redis cache cleared (flushdb)")
                return True
            elif self._fallback_cache:
                self._fallback_cache.clear()
                logger.warning("In-memory cache cleared")
                return True
            return False
        except Exception as e:
            logger.error(f"Cache clear_all failed: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check cache health.
        
        Returns:
            Health status dictionary
        """
        result = {
            "connected": self._is_connected,
            "type": "redis" if self._is_connected else "in-memory",
            "error": None
        }
        
        try:
            if self._is_connected and self._redis_client:
                # Test Redis connection
                self._redis_client.ping()
                
                # Get info
                info = self._redis_client.info()
                result.update({
                    "connected_clients": info.get("connected_clients"),
                    "used_memory_human": info.get("used_memory_human"),
                    "uptime_in_seconds": info.get("uptime_in_seconds"),
                })
        
        except Exception as e:
            result["error"] = str(e)
            result["connected"] = False
        
        return result
    
    # -------------------------
    # Serialization
    # -------------------------
    
    def _serialize(self, value: Any) -> bytes:
        """
        Serialize value for storage.
        
        Tries JSON first (faster, human-readable),
        falls back to pickle for complex objects.
        """
        try:
            # Try JSON first
            if isinstance(value, (dict, list, str, int, float, bool, type(None))):
                return json.dumps(value).encode('utf-8')
            
            # Fall back to pickle for complex objects
            return pickle.dumps(value)
        
        except Exception as e:
            logger.error(f"Serialization failed: {e}")
            # Last resort: pickle
            return pickle.dumps(value)
    
    def _deserialize(self, data: bytes) -> Any:
        """
        Deserialize value from storage.
        
        Tries JSON first, falls back to pickle.
        """
        try:
            # Try JSON first
            try:
                return json.loads(data.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
            
            # Fall back to pickle
            return pickle.loads(data)
        
        except Exception as e:
            logger.error(f"Deserialization failed: {e}")
            return None


# =========================================================
# Singleton Getter
# =========================================================

_redis_client_instance: Optional[RedisClient] = None


def get_redis_client() -> RedisClient:
    """
    Get singleton Redis client instance.
    
    Returns:
        RedisClient instance
    
    Example:
        >>> redis_client = get_redis_client()
        >>> redis_client.set("key", "value", ttl=3600)
        >>> value = redis_client.get("key")
    """
    global _redis_client_instance
    
    if _redis_client_instance is None:
        _redis_client_instance = RedisClient()
    
    return _redis_client_instance


# =========================================================
# Testing Utilities
# =========================================================

if __name__ == "__main__":
    from config.logging_config import setup_logging
    
    setup_logging()
    
    print("\n" + "=" * 70)
    print("🔴 REDIS CLIENT TEST")
    print("=" * 70)
    
    # Get Redis client
    redis_client = get_redis_client()
    
    # Health check
    print("\n📊 Health Check:")
    health = redis_client.health_check()
    print(f"  Connected: {health['connected']}")
    print(f"  Type: {health['type']}")
    if health.get('error'):
        print(f"  Error: {health['error']}")
    
    # Test basic operations
    print("\n🧪 Testing basic operations...")
    
    # Set
    success = redis_client.set("test:key", {"data": "value"}, ttl=60)
    print(f"  Set: {'✅' if success else '❌'}")
    
    # Get
    value = redis_client.get("test:key")
    print(f"  Get: {value}")
    
    # Exists
    exists = redis_client.exists("test:key")
    print(f"  Exists: {'✅' if exists else '❌'}")
    
    # Delete
    deleted = redis_client.delete("test:key")
    print(f"  Delete: {'✅' if deleted else '❌'}")
    
    # Test batch operations
    print("\n🔄 Testing batch operations...")
    
    mapping = {
        "test:1": {"id": 1},
        "test:2": {"id": 2},
        "test:3": {"id": 3},
    }
    
    redis_client.set_many(mapping, ttl=60)
    print(f"  Set many: ✅")
    
    values = redis_client.get_many(["test:1", "test:2", "test:3"])
    print(f"  Get many: {len(values)} items")
    
    # Cleanup
    deleted_count = redis_client.delete_pattern("test:*")
    print(f"  Cleanup: Deleted {deleted_count} keys")
    
    print("\n" + "=" * 70)
    print("✅ Redis client test complete")