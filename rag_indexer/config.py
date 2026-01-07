"""
RAG Indexer Configuration
"""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class IndexerConfig:
    """Configuration for the RAG indexer."""
    
    # MongoDB
    mongodb_uri: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    mongodb_database: str = os.getenv("MONGODB_DATABASE", "kahi")
    
    # OpenSearch
    opensearch_url: str = os.getenv("OPENSEARCH_URL", "http://localhost:9200")
    opensearch_user: Optional[str] = os.getenv("OPENSEARCH_USER")
    opensearch_password: Optional[str] = os.getenv("OPENSEARCH_PASSWORD")
    
    # Index configuration
    index_prefix: str = os.getenv("INDEX_PREFIX", "impactu_marie_agent")
    
    # Embedding model
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
    embedding_max_length: int = int(os.getenv("EMBEDDING_MAX_LENGTH", "8192"))
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "256"))
    
    # GPU configuration
    use_gpu: bool = os.getenv("USE_GPU", "true").lower() == "true"
    cuda_visible_devices: str = os.getenv("CUDA_VISIBLE_DEVICES", "0")
    
    # Processing
    batch_size: int = int(os.getenv("BATCH_SIZE", "100"))
    workers: int = int(os.getenv("WORKERS", "20"))
    
    # Chunking
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    def get_index_name(self, collection: str) -> str:
        """Get full index name for a collection."""
        return f"{self.index_prefix}_{collection}"
    
    def __post_init__(self):
        """Validate configuration."""
        if self.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
