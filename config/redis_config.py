"""
Redis Cache Configuration for Bookend Recommendation System.

Provides:
1. Redis client singleton with connection pooling
2. Cache operations (get/set/delete) with TTL
3. JSON serialization support
4. Batch operations
5. Health check utilities

Principles:
- Single Source of Truth: Uses config.settings.RedisSettings
- Singleton Pattern: One connection pool per application
- Error Handling: Graceful degradation on cache failures
- Performance: Connection pooling, pipelining support
"""

from typing import Optional, Any, Dict, List
from contextlib import contextmanager
import logging
import json
import pickle

import redis
from redis.connection import ConnectionPool
from redis.exceptions import (
    RedisError,
    ConnectionError,
    TimeoutError,
    ResponseError
)

from config.settings import get_settings

logger = logging.getLogger(__name__)


class RedisConnection:
    """
    Singleton Redis client manager.
    
    Ensures only one connection pool exists across the application.
    Thread-safe and supports automatic serialization.
    """
    
    _instance: Optional['RedisConnection'] = None
    _client: Optional[redis.Redis] = None
    _pool: Optional[ConnectionPool] = None
    _initialized: bool = False
    
    def __new__(cls) -> 'RedisConnection':
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize Redis client (only once due to singleton)."""
        if not self._initialized:
            self._initialize_client()
            self._initialized = True
    
    def _initialize_client(self) -> None:
        """
        Create and configure Redis client with connection pooling.
        
        Features:
        - Connection pooling for performance
        - Automatic retry on connection failure
        - Support for password authentication
        - Health check on initialization
        """
        settings = get_settings()
        redis_settings = settings.redis
        
        try:
            # Create connection pool
            self._pool = ConnectionPool(
                host=redis_settings.host,
                port=redis_settings.port,
                db=redis_settings.db,
                password=redis_settings.password,
                max_connections=redis_settings.max_connections,
                decode_responses=False,  # Handle encoding manually
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True
            )
            
            # Create Redis client
            self._client = redis.Redis(connection_pool=self._pool)
            
            # Test connection
            self._client.ping()
            
            logger.info(
                f"Redis client initialized: {redis_settings.host}:{redis_settings.port}, "
                f"db={redis_settings.db}, max_connections={redis_settings.max_connections}"
            )
        
        except ConnectionError as e:
            logger.error(
                f"Failed to connect to Redis: {e}",
                exc_info=True
            )
            raise
        
        except Exception as e:
            logger.error(
                f"Failed to initialize Redis client: {e}",
                exc_info=True
            )
            raise
    
    @property
    def client(self) -> redis.Redis:
        """Get Redis client instance."""
        if self._client is None:
            self._initialize_client()
        return self._client
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        serialize: str = "json"
    ) -> bool:
        """
        Set cache value with optional TTL.
        
        Args:
            key: Cache key
            value: Value to cache (will be serialized)
            ttl: Time-to-live in seconds (uses default if None)
            serialize: Serialization method ("json" or "pickle")
        
        Returns:
            bool: True if successful, False otherwise
        
        Example:
            >>> redis_conn = get_redis()
            >>> redis_conn.set("user:123", {"name": "Alice"}, ttl=3600)
            >>> redis_conn.set("model:v1", model_object, serialize="pickle")
        """
        settings = get_settings()
        ttl = ttl or settings.redis.ttl
        
        try:
            # Serialize value
            if serialize == "json":
                serialized = json.dumps(value).encode('utf-8')
            elif serialize == "pickle":
                serialized = pickle.dumps(value)
            else:
                logger.error(f"Unknown serialization method: {serialize}")
                return False
            
            # Set with TTL
            result = self._client.setex(
                name=key,
                time=ttl,
                value=serialized
            )
            
            logger.debug(f"Set cache: {key} (ttl={ttl}s, method={serialize})")
            return bool(result)
        
        except RedisError as e:
            logger.warning(f"Failed to set cache {key}: {e}")
            return False
        
        except Exception as e:
            logger.error(f"Unexpected error setting cache {key}: {e}", exc_info=True)
            return False
    
    def get(
        self,
        key: str,
        default: Any = None,
        serialize: str = "json"
    ) -> Any:
        """
        Get cached value.
        
        Args:
            key: Cache key
            default: Default value if key not found
            serialize: Serialization method used ("json" or "pickle")
        
        Returns:
            Cached value, or default if not found/error
        
        Example:
            >>> redis_conn = get_redis()
            >>> user = redis_conn.get("user:123", default={})
            >>> model = redis_conn.get("model:v1", serialize="pickle")
        """
        try:
            value = self._client.get(key)
            
            if value is None:
                logger.debug(f"Cache miss: {key}")
                return default
            
            # Deserialize value
            if serialize == "json":
                deserialized = json.loads(value.decode('utf-8'))
            elif serialize == "pickle":
                deserialized = pickle.loads(value)
            else:
                logger.error(f"Unknown serialization method: {serialize}")
                return default
            
            logger.debug(f"Cache hit: {key}")
            return deserialized
        
        except RedisError as e:
            logger.warning(f"Failed to get cache {key}: {e}")
            return default
        
        except (json.JSONDecodeError, pickle.UnpicklingError) as e:
            logger.warning(f"Failed to deserialize cache {key}: {e}")
            return default
        
        except Exception as e:
            logger.error(f"Unexpected error getting cache {key}: {e}", exc_info=True)
            return default
    
    def delete(self, key: str) -> bool:
        """
        Delete cached value.
        
        Args:
            key: Cache key to delete
        
        Returns:
            bool: True if deleted, False otherwise
        
        Example:
            >>> redis_conn = get_redis()
            >>> redis_conn.delete("user:123")
        """
        try:
            result = self._client.delete(key)
            
            if result > 0:
                logger.debug(f"Deleted cache: {key}")
                return True
            else:
                logger.debug(f"Cache key not found: {key}")
                return False
        
        except RedisError as e:
            logger.warning(f"Failed to delete cache {key}: {e}")
            return False
        
        except Exception as e:
            logger.error(f"Unexpected error deleting cache {key}: {e}", exc_info=True)
            return False
    
    def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key to check
        
        Returns:
            bool: True if exists, False otherwise
        
        Example:
            >>> redis_conn = get_redis()
            >>> if redis_conn.exists("user:123"):
            ...     print("User cached")
        """
        try:
            return bool(self._client.exists(key))
        
        except RedisError as e:
            logger.warning(f"Failed to check existence of {key}: {e}")
            return False
        
        except Exception as e:
            logger.error(f"Unexpected error checking {key}: {e}", exc_info=True)
            return False
    
    def get_ttl(self, key: str) -> Optional[int]:
        """
        Get remaining TTL for key.
        
        Args:
            key: Cache key
        
        Returns:
            int: Remaining seconds, or None if error/no TTL
        
        Example:
            >>> redis_conn = get_redis()
            >>> ttl = redis_conn.get_ttl("user:123")
            >>> print(f"Expires in {ttl} seconds")
        """
        try:
            ttl = self._client.ttl(key)
            
            if ttl == -2:  # Key doesn't exist
                return None
            elif ttl == -1:  # Key has no expiration
                return None
            else:
                return ttl
        
        except RedisError as e:
            logger.warning(f"Failed to get TTL for {key}: {e}")
            return None
        
        except Exception as e:
            logger.error(f"Unexpected error getting TTL for {key}: {e}", exc_info=True)
            return None
    
    def mget(
        self,
        keys: List[str],
        serialize: str = "json"
    ) -> Dict[str, Any]:
        """
        Get multiple cached values in one operation.
        
        Args:
            keys: List of cache keys
            serialize: Serialization method
        
        Returns:
            dict: Mapping of key to value (only existing keys)
        
        Example:
            >>> redis_conn = get_redis()
            >>> users = redis_conn.mget(["user:1", "user:2", "user:3"])
            >>> print(f"Found {len(users)} users")
        """
        if not keys:
            return {}
        
        try:
            values = self._client.mget(keys)
            
            result = {}
            for key, value in zip(keys, values):
                if value is None:
                    continue
                
                try:
                    # Deserialize
                    if serialize == "json":
                        result[key] = json.loads(value.decode('utf-8'))
                    elif serialize == "pickle":
                        result[key] = pickle.loads(value)
                
                except Exception as e:
                    logger.warning(f"Failed to deserialize {key}: {e}")
                    continue
            
            logger.debug(f"Batch get: {len(result)}/{len(keys)} hits")
            return result
        
        except RedisError as e:
            logger.warning(f"Failed to batch get {len(keys)} keys: {e}")
            return {}
        
        except Exception as e:
            logger.error(f"Unexpected error in batch get: {e}", exc_info=True)
            return {}
    
    def mset(
        self,
        mapping: Dict[str, Any],
        ttl: Optional[int] = None,
        serialize: str = "json"
    ) -> bool:
        """
        Set multiple cache values in one operation.
        
        Args:
            mapping: Dictionary of key-value pairs
            ttl: Time-to-live in seconds
            serialize: Serialization method
        
        Returns:
            bool: True if all successful
        
        Example:
            >>> redis_conn = get_redis()
            >>> redis_conn.mset({
            ...     "user:1": {"name": "Alice"},
            ...     "user:2": {"name": "Bob"}
            ... }, ttl=3600)
        """
        if not mapping:
            return True
        
        settings = get_settings()
        ttl = ttl or settings.redis.ttl
        
        try:
            # Use pipeline for atomic operation
            pipe = self._client.pipeline()
            
            for key, value in mapping.items():
                try:
                    # Serialize
                    if serialize == "json":
                        serialized = json.dumps(value).encode('utf-8')
                    elif serialize == "pickle":
                        serialized = pickle.dumps(value)
                    else:
                        continue
                    
                    pipe.setex(key, ttl, serialized)
                
                except Exception as e:
                    logger.warning(f"Failed to serialize {key}: {e}")
                    continue
            
            pipe.execute()
            
            logger.debug(f"Batch set: {len(mapping)} keys (ttl={ttl}s)")
            return True
        
        except RedisError as e:
            logger.warning(f"Failed to batch set {len(mapping)} keys: {e}")
            return False
        
        except Exception as e:
            logger.error(f"Unexpected error in batch set: {e}", exc_info=True)
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching pattern.
        
        Args:
            pattern: Redis key pattern (e.g., "user:*")
        
        Returns:
            int: Number of keys deleted
        
        Example:
            >>> redis_conn = get_redis()
            >>> deleted = redis_conn.clear_pattern("temp:*")
            >>> print(f"Deleted {deleted} temporary keys")
        
        Warning:
            Use carefully in production - can be slow with many keys
        """
        try:
            # Find matching keys
            keys = list(self._client.scan_iter(match=pattern, count=100))
            
            if not keys:
                logger.debug(f"No keys found matching pattern: {pattern}")
                return 0
            
            # Delete in batches
            deleted = self._client.delete(*keys)
            
            logger.info(f"Deleted {deleted} keys matching pattern: {pattern}")
            return deleted
        
        except RedisError as e:
            logger.warning(f"Failed to clear pattern {pattern}: {e}")
            return 0
        
        except Exception as e:
            logger.error(f"Unexpected error clearing pattern {pattern}: {e}", exc_info=True)
            return 0
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check Redis connection health.
        
        Returns:
            dict: Health status with connection info
        
        Example:
            >>> redis_conn = get_redis()
            >>> status = redis_conn.health_check()
            >>> if status['healthy']:
            ...     print("Redis is healthy")
        """
        result = {
            "healthy": False,
            "ping": False,
            "connected_clients": None,
            "used_memory_human": None,
            "error": None
        }
        
        try:
            # Test ping
            pong = self._client.ping()
            result["ping"] = pong
            
            # Get server info
            info = self._client.info()
            
            result.update({
                "healthy": True,
                "connected_clients": info.get("connected_clients"),
                "used_memory_human": info.get("used_memory_human")
            })
            
            logger.debug("Redis health check passed")
        
        except ConnectionError as e:
            result["error"] = f"Connection failed: {e}"
            logger.error(f"Redis health check failed: {e}", exc_info=True)
        
        except TimeoutError as e:
            result["error"] = f"Timeout: {e}"
            logger.error(f"Redis health check failed: {e}", exc_info=True)
        
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Redis health check failed: {e}", exc_info=True)
        
        return result
    
    def close(self) -> None:
        """
        Close Redis connection pool.
        
        Should be called on application shutdown.
        
        Example:
            >>> redis_conn = get_redis()
            >>> # ... use Redis ...
            >>> redis_conn.close()
        """
        try:
            if self._pool is not None:
                self._pool.disconnect()
                logger.info("Redis connection pool closed")
        
        except Exception as e:
            logger.error(f"Error closing Redis pool: {e}", exc_info=True)


# Singleton instance getter
_redis_instance: Optional[RedisConnection] = None


def get_redis() -> RedisConnection:
    """
    Get singleton Redis connection instance.
    
    Returns:
        RedisConnection: Singleton Redis connection
    
    Example:
        >>> redis_conn = get_redis()
        >>> redis_conn.set("key", "value", ttl=3600)
    """
    global _redis_instance
    if _redis_instance is None:
        _redis_instance = RedisConnection()
    return _redis_instance


@contextmanager
def redis_pipeline():
    """
    Context manager for Redis pipeline (batch operations).
    
    Yields:
        redis.client.Pipeline: Redis pipeline instance
    
    Example:
        >>> with redis_pipeline() as pipe:
        ...     pipe.set("key1", "value1")
        ...     pipe.set("key2", "value2")
        ...     pipe.execute()
    """
    redis_conn = get_redis()
    pipe = redis_conn.client.pipeline()
    
    try:
        yield pipe
    except Exception as e:
        logger.error(f"Pipeline operation error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    from config.logging_config import setup_logging
    
    setup_logging()
    
    print("\n" + "=" * 70)
    print("🔴 REDIS CONNECTION TEST")
    print("=" * 70)
    
    # Test Redis connection
    redis_conn = get_redis()
    
    # Health check
    print("\n📊 Health Check:")
    status = redis_conn.health_check()
    print(f"Healthy: {status['healthy']}")
    print(f"Ping: {status.get('ping')}")
    print(f"Connected Clients: {status.get('connected_clients')}")
    print(f"Used Memory: {status.get('used_memory_human')}")
    if status.get('error'):
        print(f"Error: {status['error']}")
    
    # Test basic operations (if healthy)
    if status['healthy']:
        print("\n🧪 Testing Basic Operations:")
        
        # Set
        redis_conn.set("test:key", {"value": "hello"}, ttl=60)
        print("✓ Set test:key")
        
        # Get
        value = redis_conn.get("test:key")
        print(f"✓ Get test:key: {value}")
        
        # Exists
        exists = redis_conn.exists("test:key")
        print(f"✓ Exists test:key: {exists}")
        
        # TTL
        ttl = redis_conn.get_ttl("test:key")
        print(f"✓ TTL test:key: {ttl}s")
        
        # Delete
        redis_conn.delete("test:key")
        print("✓ Delete test:key")
    
    print("\n✅ Redis connection test complete")