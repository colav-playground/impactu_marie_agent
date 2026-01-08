"""
Redis-backed cache implementation for distributed caching.

Provides persistent, distributed caching with TTL support.
Falls back to in-memory cache if Redis is unavailable.
"""

from typing import Any, Optional, Callable
import json
import logging
import pickle
from datetime import timedelta

try:
    import redis
    from redis.exceptions import RedisError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None
    RedisError = Exception

from marie_agent.core.cache import SimpleCache

logger = logging.getLogger(__name__)


class RedisCache:
    """
    Redis-backed cache with TTL support.
    
    Features:
    - Distributed caching across instances
    - Persistent storage
    - TTL per entry
    - Automatic fallback to in-memory
    - JSON and pickle serialization
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        password: Optional[str] = None,
        db: int = 0,
        prefix: str = "marie:",
        default_ttl: int = 300,
        fallback_to_memory: bool = True
    ):
        """
        Initialize Redis cache.
        
        Args:
            host: Redis host
            port: Redis port
            password: Redis password
            db: Redis database number
            prefix: Key prefix for namespacing
            default_ttl: Default TTL in seconds
            fallback_to_memory: Use in-memory cache if Redis fails
        """
        self.prefix = prefix
        self.default_ttl = default_ttl
        self.fallback_to_memory = fallback_to_memory
        self._redis_client: Optional[redis.Redis] = None
        self._fallback_cache: Optional[SimpleCache] = None
        self._redis_available = False
        
        # Try to connect to Redis
        if REDIS_AVAILABLE:
            try:
                self._redis_client = redis.Redis(
                    host=host,
                    port=port,
                    password=password,
                    db=db,
                    decode_responses=False,  # We'll handle encoding
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
                
                # Test connection
                self._redis_client.ping()
                self._redis_available = True
                logger.info(f"Redis cache connected: {host}:{port} (db={db})")
                
            except (RedisError, Exception) as e:
                logger.warning(f"Failed to connect to Redis: {e}")
                self._redis_available = False
        else:
            logger.warning("Redis not installed - using in-memory cache")
        
        # Setup fallback if needed
        if not self._redis_available and fallback_to_memory:
            self._fallback_cache = SimpleCache(default_ttl=default_ttl)
            logger.info("Using in-memory fallback cache")
    
    def _make_key(self, key: str) -> str:
        """Create prefixed key."""
        return f"{self.prefix}{key}"
    
    def _serialize(self, value: Any) -> bytes:
        """
        Serialize value for storage.
        
        Tries JSON first (for interoperability), falls back to pickle.
        """
        try:
            # Try JSON for simple types
            json_str = json.dumps(value)
            return b"json:" + json_str.encode('utf-8')
        except (TypeError, ValueError):
            # Use pickle for complex objects
            return b"pickle:" + pickle.dumps(value)
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        if data.startswith(b"json:"):
            return json.loads(data[5:].decode('utf-8'))
        elif data.startswith(b"pickle:"):
            return pickle.loads(data[7:])
        else:
            # Legacy format, try pickle
            return pickle.loads(data)
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        if self._redis_available and self._redis_client:
            try:
                redis_key = self._make_key(key)
                data = self._redis_client.get(redis_key)
                
                if data:
                    logger.debug(f"Redis HIT: {key[:20]}...")
                    return self._deserialize(data)
                
                logger.debug(f"Redis MISS: {key[:20]}...")
                return None
                
            except (RedisError, Exception) as e:
                logger.error(f"Redis get error: {e}")
                
                # Try fallback
                if self._fallback_cache:
                    return self._fallback_cache.get(key)
                return None
        
        # Use fallback cache
        if self._fallback_cache:
            return self._fallback_cache.get(key)
        
        return None
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: TTL in seconds (uses default if None)
            
        Returns:
            True if successful
        """
        ttl = ttl or self.default_ttl
        
        if self._redis_available and self._redis_client:
            try:
                redis_key = self._make_key(key)
                serialized = self._serialize(value)
                
                self._redis_client.setex(
                    redis_key,
                    timedelta(seconds=ttl),
                    serialized
                )
                
                logger.debug(f"Redis SET: {key[:20]}... (TTL: {ttl}s)")
                return True
                
            except (RedisError, Exception) as e:
                logger.error(f"Redis set error: {e}")
                
                # Try fallback
                if self._fallback_cache:
                    self._fallback_cache.set(key, value, ttl)
                return False
        
        # Use fallback cache
        if self._fallback_cache:
            self._fallback_cache.set(key, value, ttl)
            return True
        
        return False
    
    def delete(self, key: str) -> bool:
        """
        Delete entry from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if entry was deleted
        """
        if self._redis_available and self._redis_client:
            try:
                redis_key = self._make_key(key)
                result = self._redis_client.delete(redis_key)
                
                logger.debug(f"Redis DELETE: {key[:20]}...")
                return result > 0
                
            except (RedisError, Exception) as e:
                logger.error(f"Redis delete error: {e}")
                
                if self._fallback_cache:
                    return self._fallback_cache.delete(key)
                return False
        
        if self._fallback_cache:
            return self._fallback_cache.delete(key)
        
        return False
    
    def clear(self, pattern: Optional[str] = None) -> int:
        """
        Clear cache entries.
        
        Args:
            pattern: Optional pattern to match keys (e.g., "schema:*")
            
        Returns:
            Number of entries deleted
        """
        if self._redis_available and self._redis_client:
            try:
                if pattern:
                    # Delete matching keys
                    full_pattern = self._make_key(pattern)
                    keys = self._redis_client.keys(full_pattern)
                    if keys:
                        count = self._redis_client.delete(*keys)
                        logger.info(f"Redis cleared {count} keys matching {pattern}")
                        return count
                else:
                    # Delete all keys with our prefix
                    full_pattern = self._make_key("*")
                    keys = self._redis_client.keys(full_pattern)
                    if keys:
                        count = self._redis_client.delete(*keys)
                        logger.info(f"Redis cleared {count} keys")
                        return count
                
                return 0
                
            except (RedisError, Exception) as e:
                logger.error(f"Redis clear error: {e}")
                
                if self._fallback_cache:
                    self._fallback_cache.clear()
                return 0
        
        if self._fallback_cache:
            self._fallback_cache.clear()
            return 0
        
        return 0
    
    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], Any],
        ttl: Optional[int] = None
    ) -> Any:
        """
        Get from cache or compute and cache result.
        
        Args:
            key: Cache key
            compute_fn: Function to compute value if not cached
            ttl: TTL in seconds
            
        Returns:
            Cached or computed value
        """
        # Try to get from cache
        cached = self.get(key)
        if cached is not None:
            return cached
        
        # Compute value
        logger.debug(f"Cache MISS: Computing value for {key[:20]}...")
        value = compute_fn()
        
        # Cache result
        self.set(key, value, ttl)
        
        return value
    
    def ping(self) -> bool:
        """
        Check if Redis is available.
        
        Returns:
            True if Redis is responding
        """
        if self._redis_available and self._redis_client:
            try:
                self._redis_client.ping()
                return True
            except (RedisError, Exception):
                return False
        return False
    
    def get_stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            Statistics dict
        """
        stats = {
            "backend": "redis" if self._redis_available else "memory",
            "redis_available": self._redis_available,
            "prefix": self.prefix,
            "default_ttl": self.default_ttl
        }
        
        if self._redis_available and self._redis_client:
            try:
                info = self._redis_client.info("stats")
                stats.update({
                    "total_commands": info.get("total_commands_processed", 0),
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0)
                })
                
                # Calculate hit rate
                hits = stats["keyspace_hits"]
                misses = stats["keyspace_misses"]
                total = hits + misses
                if total > 0:
                    stats["hit_rate_percent"] = round((hits / total) * 100, 2)
                
            except (RedisError, Exception) as e:
                logger.warning(f"Failed to get Redis stats: {e}")
        
        elif self._fallback_cache:
            stats.update(self._fallback_cache.stats())
        
        return stats


# Global Redis cache instances
_redis_schema_cache: Optional[RedisCache] = None
_redis_entity_cache: Optional[RedisCache] = None
_redis_query_cache: Optional[RedisCache] = None


def get_redis_schema_cache(
    host: str = "localhost",
    port: int = 6379,
    password: Optional[str] = None
) -> RedisCache:
    """
    Get global Redis schema cache instance.
    
    Args:
        host: Redis host
        port: Redis port
        password: Redis password
        
    Returns:
        RedisCache instance
    """
    global _redis_schema_cache
    if _redis_schema_cache is None:
        _redis_schema_cache = RedisCache(
            host=host,
            port=port,
            password=password,
            db=0,
            prefix="marie:schema:",
            default_ttl=600,  # 10 minutes
            fallback_to_memory=True
        )
    return _redis_schema_cache


def get_redis_entity_cache(
    host: str = "localhost",
    port: int = 6379,
    password: Optional[str] = None
) -> RedisCache:
    """
    Get global Redis entity cache instance.
    
    Args:
        host: Redis host
        port: Redis port
        password: Redis password
        
    Returns:
        RedisCache instance
    """
    global _redis_entity_cache
    if _redis_entity_cache is None:
        _redis_entity_cache = RedisCache(
            host=host,
            port=port,
            password=password,
            db=0,
            prefix="marie:entity:",
            default_ttl=300,  # 5 minutes
            fallback_to_memory=True
        )
    return _redis_entity_cache


def get_redis_query_cache(
    host: str = "localhost",
    port: int = 6379,
    password: Optional[str] = None
) -> RedisCache:
    """
    Get global Redis query cache instance.
    
    Args:
        host: Redis host
        port: Redis port
        password: Redis password
        
    Returns:
        RedisCache instance
    """
    global _redis_query_cache
    if _redis_query_cache is None:
        _redis_query_cache = RedisCache(
            host=host,
            port=port,
            password=password,
            db=0,
            prefix="marie:query:",
            default_ttl=60,  # 1 minute
            fallback_to_memory=True
        )
    return _redis_query_cache
