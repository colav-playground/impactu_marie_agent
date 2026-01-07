#!/usr/bin/env python3
"""
RAG Indexer CLI - Command line interface for indexing KAHI documents to OpenSearch
"""
import sys
import logging
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_indexer.config import IndexerConfig
from rag_indexer.indexer import RAGIndexer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='RAG Indexer - Index KAHI MongoDB documents to OpenSearch with BGE-M3 embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index all works with default settings
  python -m rag_indexer.cli --collection works
  
  # Index first 1000 documents with GPU
  python -m rag_indexer.cli --collection works --limit 1000 --gpu
  
  # Index without deleting existing index
  python -m rag_indexer.cli --collection works --no-delete
  
  # Index with custom workers and batch size
  python -m rag_indexer.cli --collection works --workers 30 --batch-size 200
  
  # Index person collection
  python -m rag_indexer.cli --collection person --limit 10000
  
  # Index with custom MongoDB and OpenSearch
  python -m rag_indexer.cli --collection works \\
    --mongodb-uri mongodb://user:pass@localhost:27017/ \\
    --opensearch-url http://localhost:9200
        """
    )
    
    # Collection
    parser.add_argument(
        '--collection',
        type=str,
        default='works',
        choices=['works', 'person', 'affiliations', 'sources', 'projects', 'patents'],
        help='Collection to index (default: works)'
    )
    
    # Limit
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of documents to index (default: all)'
    )
    
    # Index management
    parser.add_argument(
        '--no-delete',
        action='store_true',
        help='Do not delete existing index before indexing'
    )
    
    # MongoDB
    parser.add_argument(
        '--mongodb-uri',
        type=str,
        help='MongoDB connection URI (default: from MONGODB_URI env var)'
    )
    
    parser.add_argument(
        '--mongodb-database',
        type=str,
        help='MongoDB database name (default: from MONGODB_DATABASE env var or "kahi")'
    )
    
    # OpenSearch
    parser.add_argument(
        '--opensearch-url',
        type=str,
        help='OpenSearch URL (default: from OPENSEARCH_URL env var)'
    )
    
    parser.add_argument(
        '--opensearch-user',
        type=str,
        help='OpenSearch username (default: from OPENSEARCH_USER env var)'
    )
    
    parser.add_argument(
        '--opensearch-password',
        type=str,
        help='OpenSearch password (default: from OPENSEARCH_PASSWORD env var)'
    )
    
    # Index configuration
    parser.add_argument(
        '--index-prefix',
        type=str,
        help='Index name prefix (default: from INDEX_PREFIX env var or "impactu_marie_agent")'
    )
    
    # Embedding model
    parser.add_argument(
        '--embedding-model',
        type=str,
        help='Embedding model name (default: from EMBEDDING_MODEL env var or "BAAI/bge-m3")'
    )
    
    # GPU
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Use GPU for embeddings (requires CUDA)'
    )
    
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Force CPU usage even if GPU is available'
    )
    
    parser.add_argument(
        '--cuda-devices',
        type=str,
        help='CUDA visible devices (default: from CUDA_VISIBLE_DEVICES env var or "0")'
    )
    
    # Processing
    parser.add_argument(
        '--workers',
        type=int,
        help='Number of parallel workers (default: from WORKERS env var or 20)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size for processing (default: from BATCH_SIZE env var or 100)'
    )
    
    # Verbosity
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging (DEBUG level)'
    )
    
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Quiet mode (only show warnings and errors)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Create configuration
    config = IndexerConfig()
    
    # Override with CLI arguments
    if args.mongodb_uri:
        config.mongodb_uri = args.mongodb_uri
    if args.mongodb_database:
        config.mongodb_database = args.mongodb_database
    if args.opensearch_url:
        config.opensearch_url = args.opensearch_url
    if args.opensearch_user:
        config.opensearch_user = args.opensearch_user
    if args.opensearch_password:
        config.opensearch_password = args.opensearch_password
    if args.index_prefix:
        config.index_prefix = args.index_prefix
    if args.embedding_model:
        config.embedding_model = args.embedding_model
    if args.gpu:
        config.use_gpu = True
    if args.no_gpu:
        config.use_gpu = False
    if args.cuda_devices:
        config.cuda_visible_devices = args.cuda_devices
    if args.workers:
        config.workers = args.workers
    if args.batch_size:
        config.batch_size = args.batch_size
    
    # Print configuration
    logger.info("=" * 80)
    logger.info("RAG Indexer Configuration")
    logger.info("=" * 80)
    logger.info(f"Collection: {args.collection}")
    logger.info(f"MongoDB URI: {config.mongodb_uri}")
    logger.info(f"MongoDB Database: {config.mongodb_database}")
    logger.info(f"OpenSearch URL: {config.opensearch_url}")
    logger.info(f"Index Prefix: {config.index_prefix}")
    logger.info(f"Embedding Model: {config.embedding_model}")
    logger.info(f"Embedding Dimension: {config.embedding_dimension}")
    logger.info(f"Use GPU: {config.use_gpu}")
    if config.use_gpu:
        logger.info(f"CUDA Devices: {config.cuda_visible_devices}")
    logger.info(f"Workers: {config.workers}")
    logger.info(f"Batch Size: {config.batch_size}")
    logger.info(f"Limit: {args.limit if args.limit else 'All documents'}")
    logger.info(f"Delete Index: {not args.no_delete}")
    logger.info("=" * 80)
    
    # Create indexer
    indexer = RAGIndexer(config)
    
    # Run indexing
    try:
        success = indexer.run(
            collection=args.collection,
            limit=args.limit,
            delete_index=not args.no_delete
        )
        
        if success:
            logger.info("✅ Indexing completed successfully!")
            sys.exit(0)
        else:
            logger.error("❌ Indexing failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Indexing interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"❌ Fatal error during indexing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
