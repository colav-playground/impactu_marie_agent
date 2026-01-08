"""
Simple in-memory caching system with TTL.

Provides caching for expensive operations like schema inspection,
entity resolution, and repeated queries.
"""

from typing import Any, Optional, Dict, Callable
from datetime import datetime, timedelta
from threading import Lock
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


class CacheEntry:
    """Single cache entry with TTL."""
    
    def __init__(self, value: Any, ttl_seconds: int):
        """
        Initialize cache entry.
        
        Args:
            value: Cached value
            ttl_seconds: Time to live in seconds
        """
        self.value = value
        self.created_at = datetime.now()
        self.expires_at = self.created_at + timedelta(seconds=ttl_seconds)
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        return datetime.now() > self.expires_at
    
    def time_remaining(self) -> float:
        """Get remaining time in seconds."""
        if self.is_expired():
            return 0.0
        return (self.expires_at - datetime.now()).total_seconds()


class SimpleCache:
    """
    Thread-safe in-memory cache with TTL support.
    
    Features:
    - TTL per entry
    - Thread-safe operations
    - Automatic cleanup of expired entries
    - Hit/miss statistics
    """
    
    def __init__(self, default_ttl: int = 300):
        """
        Initialize cache.
        
        Args:
            default_ttl: Default TTL in seconds (5 minutes)
        """
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = Lock()
        self.default_ttl = default_ttl
        
        # Statistics
        self._hits = 0
        self._misses = 0
        
        logger.info(f"Cache initialized with default TTL: {default_ttl}s")
    
    def _make_key(self, key: Any) -> str:
        """
        Create cache key from any hashable object.
        
        Args:
            key: Key object (string, tuple, dict, etc)
            
        Returns:
            String cache key
        """
        if isinstance(key, str):
            return key
        
        # For dicts and complex objects, serialize to JSON and hash
        try:
            serialized = json.dumps(key, sort_keys=True)
            return hashlib.md5(serialized.encode()).hexdigest()
        except (TypeError, ValueError):
            # Fallback to string representation
            return hashlib.md5(str(key).encode()).hexdigest()
    
    def get(self, key: Any) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        cache_key = self._make_key(key)
        
        with self._lock:
            entry = self._cache.get(cache_key)
            
            if entry is None:
                self._misses += 1
                return None
            
            if entry.is_expired():
                # Clean up expired entry
                del self._cache[cache_key]
                self._misses += 1
                return None
            
            self._hits += 1
            logger.debug(f"Cache HIT: {cache_key[:16]}... (TTL: {entry.time_remaining():.1f}s)")
            return entry.value
    
    def set(self, key: Any, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: TTL in seconds (uses default if None)
        """
        cache_key = self._make_key(key)
        ttl = ttl or self.default_ttl
        
        with self._lock:
            self._cache[cache_key] = CacheEntry(value, ttl)
            logger.debug(f"Cache SET: {cache_key[:16]}... (TTL: {ttl}s)")
    
    def delete(self, key: Any) -> bool:
        """
        Delete entry from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if entry was deleted
        """
        cache_key = self._make_key(key)
        
        with self._lock:
            if cache_key in self._cache:
                del self._cache[cache_key]
                logger.debug(f"Cache DELETE: {cache_key[:16]}...")
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info(f"Cache cleared: {count} entries removed")
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.
        
        Returns:
            Number of entries removed
        """
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                del self._cache[key]
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired entries")
            
            return len(expired_keys)
    
    def get_or_compute(
        self,
        key: Any,
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
        logger.debug(f"Cache MISS: Computing value...")
        value = compute_fn()
        
        # Cache result
        self.set(key, value, ttl)
        
        return value
    
    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Statistics dict
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0.0
            
            return {
                "size": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "total_requests": total,
                "hit_rate_percent": round(hit_rate, 2),
                "default_ttl_seconds": self.default_ttl
            }
    
    def __len__(self) -> int:
        """Get number of cached entries."""
        with self._lock:
            return len(self._cache)


# Global cache instances
_schema_cache: Optional[SimpleCache] = None
_entity_cache: Optional[SimpleCache] = None
_query_cache: Optional[SimpleCache] = None


def get_schema_cache() -> SimpleCache:
    """
    Get global schema cache instance.
    
    TTL: 600 seconds (10 minutes) - schemas don't change often
    
    Returns:
        Schema cache
    """
    global _schema_cache
    if _schema_cache is None:
        _schema_cache = SimpleCache(default_ttl=600)
    return _schema_cache


def get_entity_cache() -> SimpleCache:
    """
    Get global entity resolution cache instance.
    
    TTL: 300 seconds (5 minutes) - entities relatively stable
    
    Returns:
        Entity cache
    """
    global _entity_cache
    if _entity_cache is None:
        _entity_cache = SimpleCache(default_ttl=300)
    return _entity_cache


def get_query_cache() -> SimpleCache:
    """
    Get global query result cache instance.
    
    TTL: 60 seconds (1 minute) - queries can change frequently
    
    Returns:
        Query cache
    """
    global _query_cache
    if _query_cache is None:
        _query_cache = SimpleCache(default_ttl=60)
    return _query_cache


def clear_all_caches() -> None:
    """Clear all global caches."""
    if _schema_cache:
        _schema_cache.clear()
    if _entity_cache:
        _entity_cache.clear()
    if _query_cache:
        _query_cache.clear()
    logger.info("All caches cleared")


def get_all_cache_stats() -> Dict[str, Dict[str, Any]]:
    """
    Get statistics for all caches.
    
    Returns:
        Dict of cache stats
    """
    stats = {}
    
    if _schema_cache:
        stats["schema_cache"] = _schema_cache.stats()
    if _entity_cache:
        stats["entity_cache"] = _entity_cache.stats()
    if _query_cache:
        stats["query_cache"] = _query_cache.stats()
    
    return stats
