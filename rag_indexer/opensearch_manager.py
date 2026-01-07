"""
OpenSearch index management
"""
import logging
from typing import Dict, Any, Optional
from opensearchpy import OpenSearch, helpers
from opensearchpy.exceptions import RequestError

logger = logging.getLogger(__name__)


class OpenSearchIndexManager:
    """Manage OpenSearch indices with K-NN configuration."""
    
    def __init__(self, opensearch_url: str, 
                 user: Optional[str] = None,
                 password: Optional[str] = None,
                 embedding_dimension: int = 1024):
        """
        Initialize OpenSearch connection.
        
        Args:
            opensearch_url: OpenSearch cluster URL
            user: Optional username for authentication
            password: Optional password for authentication
            embedding_dimension: Dimension of embedding vectors
        """
        auth = (user, password) if user and password else None
        
        self.client = OpenSearch(
            [opensearch_url],
            http_auth=auth,
            use_ssl=False
        )
        self.embedding_dimension = embedding_dimension
    
    def ping(self) -> bool:
        """Check if OpenSearch is reachable."""
        try:
            return self.client.ping()
        except Exception as e:
            logger.error(f"Failed to ping OpenSearch: {e}")
            return False
    
    def create_index(self, index_name: str, delete_if_exists: bool = False) -> bool:
        """
        Create index with K-NN configuration for BGE-M3 embeddings.
        
        Args:
            index_name: Name of the index to create
            delete_if_exists: Whether to delete existing index
            
        Returns:
            True if successful
        """
        try:
            # Delete existing index if requested
            if delete_if_exists and self.client.indices.exists(index=index_name):
                logger.info(f"Deleting existing index: {index_name}")
                self.client.indices.delete(index=index_name)
            
            # Index configuration
            index_body = {
                "settings": {
                    "number_of_shards": 8,
                    "number_of_replicas": 1,
                    "refresh_interval": "30s",
                    "index.knn": True,
                    "index.knn.algo_param.ef_search": 128
                },
                "mappings": {
                    "properties": {
                        "id": {
                            "type": "keyword"
                        },
                        "embedding": {
                            "type": "knn_vector",
                            "dimension": self.embedding_dimension,
                            "method": {
                                "name": "hnsw",
                                "space_type": "cosinesimil",
                                "engine": "faiss",
                                "parameters": {
                                    "ef_construction": 256,
                                    "m": 32
                                }
                            }
                        },
                        "text": {
                            "type": "text",
                            "analyzer": "standard"
                        },
                        "title": {
                            "type": "text",
                            "analyzer": "standard"
                        },
                        "abstract": {
                            "type": "text",
                            "analyzer": "standard"
                        },
                        "keywords": {
                            "type": "keyword"
                        },
                        "authors": {
                            "type": "text"
                        },
                        "year": {
                            "type": "integer"
                        },
                        "citations_count": {
                            "type": "integer"
                        },
                        "work_id": {
                            "type": "keyword"
                        },
                        "doi": {
                            "type": "keyword"
                        },
                        "url": {
                            "type": "keyword"
                        },
                        "source": {
                            "type": "keyword"
                        },
                        "type": {
                            "type": "keyword"
                        }
                    }
                }
            }
            
            self.client.indices.create(index=index_name, body=index_body)
            logger.info(f"Created index: {index_name}")
            return True
            
        except RequestError as e:
            if e.error == 'resource_already_exists_exception':
                logger.info(f"Index {index_name} already exists")
                return True
            else:
                logger.error(f"Error creating index: {e}")
                return False
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            return False
    
    def bulk_index(self, index_name: str, documents: list) -> tuple[int, int]:
        """
        Bulk index documents to OpenSearch.
        
        Args:
            index_name: Name of the index
            documents: List of documents to index
            
        Returns:
            Tuple of (success_count, error_count)
        """
        try:
            actions = [
                {
                    "_index": index_name,
                    "_id": doc.get('id'),
                    "_source": doc
                }
                for doc in documents
            ]
            
            success, errors = helpers.bulk(
                self.client,
                actions,
                raise_on_error=False,
                raise_on_exception=False
            )
            
            return success, len(errors) if errors else 0
            
        except Exception as e:
            logger.error(f"Error during bulk indexing: {e}")
            return 0, len(documents)
    
    def get_index_stats(self, index_name: str) -> Optional[Dict[str, Any]]:
        """Get statistics for an index."""
        try:
            stats = self.client.indices.stats(index=index_name)
            return stats
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return None
    
    def get_document_count(self, index_name: str) -> int:
        """Get number of documents in an index."""
        try:
            count = self.client.count(index=index_name)
            return count['count']
        except Exception as e:
            logger.error(f"Error getting document count: {e}")
            return 0
