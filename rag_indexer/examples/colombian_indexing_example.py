"""
Example: How the multi-agent system can use Colombian entity filtering.

This script demonstrates internal usage - NOT for CLI.
"""
import asyncio
import logging
from rag_indexer import (
    index_colombian_entities,
    setup_colombian_indices,
    RAGIndexer,
    IndexerConfig,
    is_colombian_work,
    is_colombian_person,
    is_colombian_affiliation
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Example 1: Create Colombian-specific indices (separate from main indices)
# ============================================================================

def example_create_colombian_indices():
    """Create separate indices for Colombian entities only."""
    logger.info("Example 1: Creating Colombian-specific indices")
    logger.info("=" * 80)
    
    # This creates:
    # - impactu_marie_agent_affiliations_colombia
    # - impactu_marie_agent_person_colombia  
    # - impactu_marie_agent_works_colombia
    
    success = setup_colombian_indices(
        limit=1000,  # For testing, use small sample
        gpu=True
    )
    
    if success:
        logger.info("✅ Colombian indices created successfully")
        logger.info("These are SEPARATE from main indices")
    else:
        logger.error("❌ Failed to create Colombian indices")
    
    return success


# ============================================================================
# Example 2: Index single Colombian collection
# ============================================================================

def example_index_colombian_persons():
    """Index only Colombian persons."""
    logger.info("Example 2: Indexing Colombian persons only")
    logger.info("=" * 80)
    
    success = index_colombian_entities(
        collection='person',
        limit=5000,
        gpu=True
    )
    
    if success:
        logger.info("✅ Colombian persons indexed")
    else:
        logger.error("❌ Failed to index Colombian persons")
    
    return success


# ============================================================================
# Example 3: Using RAGIndexer directly with filter
# ============================================================================

def example_direct_indexer_usage():
    """Use RAGIndexer directly with Colombian filter."""
    logger.info("Example 3: Direct RAGIndexer usage with filter")
    logger.info("=" * 80)
    
    # Create config
    config = IndexerConfig()
    config.use_gpu = True
    config.workers = 10
    config.batch_size = 50
    
    # Create indexer
    indexer = RAGIndexer(config)
    
    # Index Colombian works
    success = indexer.run(
        collection='works',
        limit=1000,
        delete_index=True,
        colombia_only=True,      # Enable Colombian filter
        index_suffix='_colombia'  # Use separate index
    )
    
    if success:
        logger.info("✅ Colombian works indexed")
    else:
        logger.error("❌ Failed to index Colombian works")
    
    return success


# ============================================================================
# Example 4: Checking if entity is Colombian (for validation)
# ============================================================================

def example_check_colombian_entity():
    """Check if entities are Colombian."""
    logger.info("Example 4: Checking Colombian entities")
    logger.info("=" * 80)
    
    # Example work document
    work = {
        'authors': [
            {
                'affiliations': [
                    {
                        'addresses': [
                            {'country_code': 'CO'}
                        ]
                    }
                ]
            }
        ]
    }
    
    if is_colombian_work(work):
        logger.info("✅ Work has Colombian connection")
    else:
        logger.info("❌ Work is not Colombian")
    
    # Example person document
    person = {
        'affiliations': [
            {
                'addresses': [
                    {'country_code': 'CO'}
                ]
            }
        ]
    }
    
    if is_colombian_person(person):
        logger.info("✅ Person has Colombian affiliation")
    else:
        logger.info("❌ Person is not Colombian")
    
    # Example affiliation document
    affiliation = {
        'addresses': [
            {'country_code': 'CO'}
        ]
    }
    
    if is_colombian_affiliation(affiliation):
        logger.info("✅ Affiliation is Colombian")
    else:
        logger.info("❌ Affiliation is not Colombian")


# ============================================================================
# Example 5: Agent integration
# ============================================================================

class ColombianResearchAgent:
    """Example agent that works with Colombian data."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def initialize(self):
        """Initialize agent and ensure Colombian indices exist."""
        self.logger.info("Initializing Colombian Research Agent")
        
        # Ensure Colombian-specific indices exist
        success = setup_colombian_indices(limit=None, gpu=True)
        
        if success:
            self.logger.info("✅ Colombian indices ready")
        else:
            self.logger.warning("⚠️ Colombian indices may not be available")
        
        return success
    
    async def search_colombian_authors(self, query: str):
        """
        Search only Colombian authors.
        
        Uses the specialized _colombia index for faster, focused searches.
        """
        index_name = "impactu_marie_agent_person_colombia"
        
        self.logger.info(f"Searching Colombian authors: {query}")
        self.logger.info(f"Using index: {index_name}")
        
        # Here you would use OpenSearch client to search
        # This is just a placeholder showing the concept
        
        return {
            'index': index_name,
            'query': query,
            'results': []
        }
    
    async def update_colombian_indices(self):
        """Update Colombian-specific indices."""
        self.logger.info("Updating Colombian indices")
        
        # Re-index with fresh data
        success = setup_colombian_indices(limit=None, gpu=True)
        
        if success:
            self.logger.info("✅ Colombian indices updated")
        else:
            self.logger.error("❌ Failed to update Colombian indices")
        
        return success


async def example_agent_usage():
    """Example of agent using Colombian indices."""
    logger.info("Example 5: Agent usage")
    logger.info("=" * 80)
    
    agent = ColombianResearchAgent()
    
    # Initialize agent (creates Colombian indices if needed)
    await agent.initialize()
    
    # Search Colombian authors
    results = await agent.search_colombian_authors("machine learning")
    logger.info(f"Search results: {results}")
    
    # Update indices periodically
    await agent.update_colombian_indices()


# ============================================================================
# Main
# ============================================================================

def main():
    """Run examples."""
    logger.info("=" * 80)
    logger.info("Colombian Entity Indexing Examples")
    logger.info("These are for internal multi-agent system use only")
    logger.info("=" * 80)
    logger.info("")
    
    # Uncomment the example you want to run:
    
    # example_create_colombian_indices()
    # example_index_colombian_persons()
    # example_direct_indexer_usage()
    # example_check_colombian_entity()
    
    # For async example:
    # asyncio.run(example_agent_usage())
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("Examples completed")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
