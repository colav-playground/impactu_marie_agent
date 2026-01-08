"""
Colombian Entity Indexer - Specialized indexer for Colombian entities only.

This module is used internally by the multi-agent system to create
separate indices for Colombian-specific data.

NOT exposed via CLI - only for programmatic use by agents.
"""
import logging
from typing import Optional

from .config import IndexerConfig
from .indexer import RAGIndexer

logger = logging.getLogger(__name__)


def index_colombian_entities(
    collection: str = 'works',
    limit: Optional[int] = None,
    gpu: bool = True
) -> bool:
    """
    Index only Colombian entities to separate indices.
    
    This creates indices with '_colombia' suffix containing only
    Colombian works, persons, or affiliations.
    
    Args:
        collection: Collection to index (works, person, affiliations)
        limit: Maximum documents to process (None = all)
        gpu: Use GPU acceleration
        
    Returns:
        True if successful
        
    Example:
        >>> # Called by agent internally
        >>> success = index_colombian_entities('person', limit=10000)
        >>> # Creates index: impactu_marie_agent_person_colombia
    """
    logger.info("=" * 80)
    logger.info("Indexing Colombian Entities")
    logger.info(f"Collection: {collection}")
    logger.info("=" * 80)
    
    # Create config
    config = IndexerConfig()
    config.use_gpu = gpu
    
    # Create indexer
    indexer = RAGIndexer(config)
    
    # Index with Colombian filter
    success = indexer.run(
        collection=collection,
        limit=limit,
        delete_index=True,
        colombia_only=True,  # Enable Colombian filter
        index_suffix='_colombia'  # Separate index
    )
    
    return success


def setup_colombian_indices(
    limit: Optional[int] = None,
    gpu: bool = True
) -> bool:
    """
    Setup all Colombian-specific indices.
    
    Creates separate indices for:
    - Colombian affiliations
    - Colombian persons
    - Colombian works
    
    Args:
        limit: Maximum documents per collection
        gpu: Use GPU acceleration
        
    Returns:
        True if all successful
    """
    logger.info("=" * 80)
    logger.info("Setting up Colombian-specific indices")
    logger.info("=" * 80)
    
    collections = ['affiliations', 'person', 'works']
    results = []
    
    for collection in collections:
        logger.info("")
        logger.info(f"📍 Indexing Colombian {collection}...")
        
        try:
            success = index_colombian_entities(
                collection=collection,
                limit=limit,
                gpu=gpu
            )
            results.append(success)
            
            if success:
                logger.info(f"✅ Colombian {collection} indexed")
            else:
                logger.error(f"❌ Failed to index Colombian {collection}")
                
        except Exception as e:
            logger.exception(f"❌ Error indexing Colombian {collection}: {e}")
            results.append(False)
    
    logger.info("")
    logger.info("=" * 80)
    
    if all(results):
        logger.info("✅ All Colombian indices created successfully!")
        return True
    else:
        logger.warning("⚠️ Some Colombian indices failed")
        return False
