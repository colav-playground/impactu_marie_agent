"""
OpenSearch connection manager with singleton pattern.

Provides centralized, reusable OpenSearch client connections.
"""

from typing import Optional
import logging
from opensearchpy import OpenSearch
from threading import Lock

from marie_agent.config import config, SystemConstants
from marie_agent.core.exceptions import OpenSearchError

logger = logging.getLogger(__name__)


class OpenSearchManager:
    """
    Singleton manager for OpenSearch connections.
    
    Ensures only one client instance exists and is reused across the application.
    Thread-safe implementation.
    """
    
    _instance: Optional['OpenSearchManager'] = None
    _lock: Lock = Lock()
    _client: Optional[OpenSearch] = None
    
    def __new__(cls):
        """Ensure singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize manager (called once per singleton)."""
        if self._client is None:
            self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize OpenSearch client."""
        try:
            self._client = OpenSearch(
                hosts=[config.opensearch.url],
                timeout=SystemConstants.OPENSEARCH_TIMEOUT,
                http_compress=True,
                use_ssl=False,
                verify_certs=False,
                ssl_assert_hostname=False,
                ssl_show_warn=False,
                max_retries=3,
                retry_on_timeout=True
            )
            
            # Test connection
            self._client.cluster.health()
            
            logger.info(f"✓ OpenSearch client initialized: {config.opensearch.url}")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenSearch client: {e}")
            raise OpenSearchError(
                f"Cannot connect to OpenSearch at {config.opensearch.url}",
                details={"error": str(e)}
            )
    
    @property
    def client(self) -> OpenSearch:
        """
        Get OpenSearch client instance.
        
        Returns:
            OpenSearch client
            
        Raises:
            OpenSearchError: If client not initialized
        """
        if self._client is None:
            raise OpenSearchError("OpenSearch client not initialized")
        
        return self._client
    
    def test_connection(self) -> bool:
        """
        Test OpenSearch connection.
        
        Returns:
            True if connection is healthy
        """
        try:
            health = self._client.cluster.health()
            return health.get("status") in ["green", "yellow"]
        except Exception as e:
            logger.error(f"OpenSearch health check failed: {e}")
            return False
    
    def close(self) -> None:
        """Close OpenSearch connection."""
        if self._client:
            try:
                self._client.close()
                logger.info("OpenSearch client closed")
            except Exception as e:
                logger.warning(f"Error closing OpenSearch client: {e}")
            finally:
                self._client = None


# Global instance
_manager: Optional[OpenSearchManager] = None
_manager_lock: Lock = Lock()


def get_opensearch_client() -> OpenSearch:
    """
    Get global OpenSearch client instance.
    
    Returns:
        OpenSearch client
        
    Raises:
        OpenSearchError: If connection fails
    """
    global _manager
    
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = OpenSearchManager()
    
    return _manager.client


def test_opensearch_connection() -> bool:
    """
    Test OpenSearch connection.
    
    Returns:
        True if connection is healthy
    """
    try:
        client = get_opensearch_client()
        health = client.cluster.health()
        return health.get("status") in ["green", "yellow"]
    except Exception as e:
        logger.error(f"OpenSearch connection test failed: {e}")
        return False
