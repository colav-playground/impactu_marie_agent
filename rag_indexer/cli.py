#!/usr/bin/env python3
"""
RAG Indexer CLI - Command line interface for indexing KAHI documents to OpenSearch
"""
import sys
import logging
import argparse
from pathlib import Path
from typing import Optional

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


def test_connections():
    """Test connections to MongoDB and OpenSearch."""
    from pymongo import MongoClient
    from opensearchpy import OpenSearch
    
    print("=" * 60)
    print("RAG Indexer - Connection Test")
    print("=" * 60)
    print()
    
    config = IndexerConfig()
    
    # Test MongoDB
    try:
        print("Testing MongoDB connection...")
        mongo = MongoClient(config.mongodb_uri)
        db = mongo[config.mongodb_database]
        collections = db.list_collection_names()
        print(f"✅ MongoDB connected: {config.mongodb_database}")
        print(f"   Collections: {', '.join(collections[:5])}")
        
        # Show document counts
        print("\nDocument counts:")
        for coll in ['affiliations', 'person', 'works']:
            if coll in collections:
                count = db[coll].count_documents({})
                print(f"   {coll}: {count:,}")
        
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        return False
    
    print()
    
    # Test OpenSearch
    try:
        print("Testing OpenSearch connection...")
        os_client = OpenSearch([config.opensearch_url])
        info = os_client.info()
        print(f"✅ OpenSearch connected: {info['cluster_name']}")
        print(f"   Version: {info['version']['number']}")
        
        # Show indices
        try:
            indices = os_client.cat.indices(format='json')
            if indices:
                print("\nCurrent indices:")
                for idx in indices:
                    print(f"   {idx['index']}: {idx['docs.count']} documents")
            else:
                print("\n   No indices found")
        except:
            print("\n   No indices found")
            
    except Exception as e:
        print(f"❌ OpenSearch connection failed: {e}")
        return False
    
    print()
    print("=" * 60)
    print("✅ All connections successful!")
    print("=" * 60)
    return True


def setup_all_indices(limit: Optional[int] = None, gpu: bool = True):
    """
    Setup all OpenSearch indices with Colombian data.
    
    Args:
        limit: Maximum documents per collection (None = all)
        gpu: Use GPU acceleration
    """
    logger.info("=" * 80)
    logger.info("RAG Indexer - Setup All Indices")
    logger.info("=" * 80)
    
    config = IndexerConfig()
    config.use_gpu = gpu
    
    collections = [
        ('affiliations', limit or 1000),
        ('person', limit or 5000),
        ('works', limit)
    ]
    
    for collection_name, coll_limit in collections:
        logger.info("")
        logger.info(f"📚 Indexing {collection_name}...")
        logger.info("-" * 80)
        
        try:
            indexer = RAGIndexer(config)
            success = indexer.run(
                collection=collection_name,
                limit=coll_limit,
                delete_index=True
            )
            
            if success:
                logger.info(f"✅ {collection_name} indexed successfully")
            else:
                logger.error(f"❌ Failed to index {collection_name}")
                
        except Exception as e:
            logger.error(f"❌ Error indexing {collection_name}: {e}")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("🎉 All indices setup completed!")
    logger.info("=" * 80)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='RAG Indexer - Index KAHI MongoDB documents to OpenSearch with BGE-M3 embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  test      Test connections to MongoDB and OpenSearch
  setup     Setup all indices with default samples
  sample    Index a sample from each collection with custom limit
  index     Index a specific collection (default)
  
Examples:
  # Test connections
  python -m rag_indexer.cli test
  
  # Setup all indices with default samples
  python -m rag_indexer.cli setup
  
  # Index sample from all collections
  python -m rag_indexer.cli sample --limit 100
  
  # Index sample with specific collections
  python -m rag_indexer.cli sample --limit 500 --collections works person affiliations
  
  # Index sample with custom batch size and workers
  python -m rag_indexer.cli sample --limit 200 --batch-size 50 --workers 10
  
  # Setup all indices with custom limit
  python -m rag_indexer.cli setup --limit 10000
  
  # Index all works with default settings
  python -m rag_indexer.cli index --collection works
  
  # Index first 1000 documents with GPU
  python -m rag_indexer.cli index --collection works --limit 1000 --gpu
  
  # Index without deleting existing index
  python -m rag_indexer.cli index --collection works --no-delete
  
  # Index with custom workers and batch size
  python -m rag_indexer.cli index --collection works --workers 30 --batch-size 200
  
  # Index person collection
  python -m rag_indexer.cli index --collection person --limit 10000
        """
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test database connections')
    
    # Sample command - Index samples from all collections
    sample_parser = subparsers.add_parser('sample', help='Index sample from each collection')
    sample_parser.add_argument(
        '--limit',
        type=int,
        required=True,
        help='Number of documents to index per collection'
    )
    sample_parser.add_argument(
        '--collections',
        nargs='+',
        default=['affiliations', 'person', 'works'],
        choices=['works', 'person', 'affiliations', 'sources', 'projects', 'patents'],
        help='Collections to sample (default: affiliations person works)'
    )
    sample_parser.add_argument(
        '--gpu',
        action='store_true',
        default=True,
        help='Use GPU for embeddings (default: True)'
    )
    sample_parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU usage'
    )
    sample_parser.add_argument(
        '--workers',
        type=int,
        help='Number of parallel workers (default: 20)'
    )
    sample_parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size for processing (default: 100)'
    )
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup all indices with sample data')
    setup_parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum documents per collection (default: affiliations=1000, person=5000, works=None)'
    )
    setup_parser.add_argument(
        '--gpu',
        action='store_true',
        default=True,
        help='Use GPU for embeddings (default: True)'
    )
    setup_parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU usage'
    )
    
    # Index command (default)
    index_parser = subparsers.add_parser('index', help='Index a specific collection')
    
    # Collection
    index_parser.add_argument(
        '--collection',
        type=str,
        default='works',
        choices=['works', 'person', 'affiliations', 'sources', 'projects', 'patents'],
        help='Collection to index (default: works)'
    )
    
    # Limit
    index_parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of documents to index (default: all)'
    )
    
    # Index management
    index_parser.add_argument(
        '--no-delete',
        action='store_true',
        help='Do not delete existing index before indexing'
    )
    
    # MongoDB
    index_parser.add_argument(
        '--mongodb-uri',
        type=str,
        help='MongoDB connection URI (default: from MONGODB_URI env var)'
    )
    
    index_parser.add_argument(
        '--mongodb-database',
        type=str,
        help='MongoDB database name (default: from MONGODB_DATABASE env var or "kahi")'
    )
    
    # OpenSearch
    index_parser.add_argument(
        '--opensearch-url',
        type=str,
        help='OpenSearch URL (default: from OPENSEARCH_URL env var)'
    )
    
    index_parser.add_argument(
        '--opensearch-user',
        type=str,
        help='OpenSearch username (default: from OPENSEARCH_USER env var)'
    )
    
    index_parser.add_argument(
        '--opensearch-password',
        type=str,
        help='OpenSearch password (default: from OPENSEARCH_PASSWORD env var)'
    )
    
    # Index configuration
    index_parser.add_argument(
        '--index-prefix',
        type=str,
        help='Index name prefix (default: from INDEX_PREFIX env var or "impactu_marie_agent")'
    )
    
    # Embedding model
    index_parser.add_argument(
        '--embedding-model',
        type=str,
        help='Embedding model name (default: from EMBEDDING_MODEL env var or "BAAI/bge-m3")'
    )
    
    # GPU
    index_parser.add_argument(
        '--gpu',
        action='store_true',
        help='Use GPU for embeddings (requires CUDA)'
    )
    
    index_parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Force CPU usage even if GPU is available'
    )
    
    index_parser.add_argument(
        '--cuda-devices',
        type=str,
        help='CUDA visible devices (default: from CUDA_VISIBLE_DEVICES env var or "0")'
    )
    
    # Processing
    index_parser.add_argument(
        '--workers',
        type=int,
        help='Number of parallel workers (default: from WORKERS env var or 20)'
    )
    
    index_parser.add_argument(
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
    
    # Default to index command if no subcommand given
    args = parser.parse_args()
    if args.command is None:
        # Backward compatibility: treat old-style args as index command
        parser.print_help()
        sys.exit(0)
    
    return args


def main():
    """Main entry point."""
    args = parse_args()
    
    # Set logging level - Default to INFO to see progress
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    else:
        logging.getLogger().setLevel(logging.INFO)
    
    # Handle commands
    if args.command == 'test':
        success = test_connections()
        sys.exit(0 if success else 1)
    
    elif args.command == 'sample':
        gpu = args.gpu and not args.no_gpu
        
        logger.info("=" * 80)
        logger.info("Indexing samples from all collections")
        logger.info(f"Limit per collection: {args.limit}")
        logger.info(f"Collections: {', '.join(args.collections)}")
        logger.info("=" * 80)
        
        config = IndexerConfig()
        config.use_gpu = gpu
        
        # Apply custom settings if provided
        if args.workers:
            config.workers = args.workers
            logger.info(f"Workers: {args.workers}")
        if args.batch_size:
            config.batch_size = args.batch_size
            logger.info(f"Batch size: {args.batch_size}")
        
        try:
            # Create indexer ONCE and reuse for all collections
            logger.info("")
            logger.info("Initializing indexer (loading model once)...")
            indexer = RAGIndexer(config)
            
            # Initialize connections and model once
            if not indexer.connect_opensearch():
                logger.error("❌ Failed to connect to OpenSearch")
                sys.exit(1)
            
            if not indexer.connect_mongodb():
                logger.error("❌ Failed to connect to MongoDB")
                sys.exit(1)
            
            if not indexer.initialize_embeddings():
                logger.error("❌ Failed to initialize embeddings")
                sys.exit(1)
            
            logger.info("✅ Indexer ready - model loaded in memory")
            logger.info("=" * 80)
            
            # Now index all collections with the same indexer
            for collection_name in args.collections:
                logger.info("")
                logger.info(f"📚 Indexing {args.limit} documents from {collection_name}...")
                logger.info("-" * 80)
                
                # Use index_works directly instead of run()
                stats = indexer.index_works(
                    collection_name=collection_name,
                    limit=args.limit,
                    delete_index=True
                )
                
                if stats['indexed'] > 0:
                    logger.info(f"✅ {collection_name} sample indexed successfully")
                else:
                    logger.error(f"❌ Failed to index {collection_name} sample")
            
            logger.info("")
            logger.info("=" * 80)
            logger.info("🎉 All samples indexed!")
            logger.info("=" * 80)
            sys.exit(0)
            
        except Exception as e:
            logger.exception(f"❌ Sample indexing failed: {e}")
            sys.exit(1)
    
    elif args.command == 'setup':
        gpu = args.gpu and not args.no_gpu
        try:
            setup_all_indices(limit=args.limit, gpu=gpu)
            sys.exit(0)
        except Exception as e:
            logger.exception(f"❌ Setup failed: {e}")
            sys.exit(1)
    
    elif args.command == 'index':
        # Create configuration
        config = IndexerConfig()
        
        # Override with CLI arguments
        if hasattr(args, 'mongodb_uri') and args.mongodb_uri:
            config.mongodb_uri = args.mongodb_uri
        if hasattr(args, 'mongodb_database') and args.mongodb_database:
            config.mongodb_database = args.mongodb_database
        if hasattr(args, 'opensearch_url') and args.opensearch_url:
            config.opensearch_url = args.opensearch_url
        if hasattr(args, 'opensearch_user') and args.opensearch_user:
            config.opensearch_user = args.opensearch_user
        if hasattr(args, 'opensearch_password') and args.opensearch_password:
            config.opensearch_password = args.opensearch_password
        if hasattr(args, 'index_prefix') and args.index_prefix:
            config.index_prefix = args.index_prefix
        if hasattr(args, 'embedding_model') and args.embedding_model:
            config.embedding_model = args.embedding_model
        if hasattr(args, 'gpu') and args.gpu:
            config.use_gpu = True
        if hasattr(args, 'no_gpu') and args.no_gpu:
            config.use_gpu = False
        if hasattr(args, 'cuda_devices') and args.cuda_devices:
            config.cuda_visible_devices = args.cuda_devices
        if hasattr(args, 'workers') and args.workers:
            config.workers = args.workers
        if hasattr(args, 'batch_size') and args.batch_size:
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
