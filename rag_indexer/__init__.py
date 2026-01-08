"""RAG Indexer for KAHI database to OpenSearch with BGE-M3 embeddings."""

__version__ = "1.0.0"

# Public API for multi-agent system
from .config import IndexerConfig
from .indexer import RAGIndexer
from .colombia_indexer import (
    index_colombian_entities,
    setup_colombian_indices
)
from .text_processing import (
    is_colombian_work,
    is_colombian_person,
    is_colombian_affiliation
)

__all__ = [
    'IndexerConfig',
    'RAGIndexer',
    'index_colombian_entities',
    'setup_colombian_indices',
    'is_colombian_work',
    'is_colombian_person',
    'is_colombian_affiliation'
]
