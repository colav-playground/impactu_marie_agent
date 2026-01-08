"""
Main indexer implementation with parallel processing
"""
import time
import logging
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

from .config import IndexerConfig
from .opensearch_manager import OpenSearchIndexManager
from .text_processing import (
    extract_work_text,
    create_work_metadata,
    should_index_work
)

logger = logging.getLogger(__name__)


class RAGIndexer:
    """Index KAHI documents to OpenSearch with embeddings."""
    
    def __init__(self, config: IndexerConfig):
        """
        Initialize the RAG indexer.
        
        Args:
            config: Indexer configuration
        """
        self.config = config
        self.mongo_client: Optional[MongoClient] = None
        self.opensearch_manager: Optional[OpenSearchIndexManager] = None
        self.embedding_model: Optional[SentenceTransformer] = None
        
    def connect_mongodb(self) -> bool:
        """Connect to MongoDB."""
        logger.info(f"Connecting to MongoDB: {self.config.mongodb_uri}")
        max_retries = 5
        
        for attempt in range(max_retries):
            try:
                self.mongo_client = MongoClient(
                    self.config.mongodb_uri,
                    serverSelectionTimeoutMS=5000
                )
                self.mongo_client.admin.command('ping')
                logger.info("✅ Successfully connected to MongoDB")
                return True
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
        
        logger.error("❌ Could not connect to MongoDB")
        return False
    
    def connect_opensearch(self) -> bool:
        """Connect to OpenSearch."""
        logger.info(f"Connecting to OpenSearch: {self.config.opensearch_url}")
        max_retries = 10
        
        for attempt in range(max_retries):
            try:
                self.opensearch_manager = OpenSearchIndexManager(
                    opensearch_url=self.config.opensearch_url,
                    user=self.config.opensearch_user,
                    password=self.config.opensearch_password,
                    embedding_dimension=self.config.embedding_dimension
                )
                
                if self.opensearch_manager.ping():
                    logger.info("✅ Successfully connected to OpenSearch")
                    return True
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
        
        logger.error("❌ Could not connect to OpenSearch")
        return False
    
    def initialize_embeddings(self) -> bool:
        """Initialize the embedding model."""
        device = 'cuda' if self.config.use_gpu and torch.cuda.is_available() else 'cpu'
        logger.info(f"Loading embedding model: {self.config.embedding_model}")
        logger.info(f"Device: {device.upper()}")
        
        try:
            self.embedding_model = SentenceTransformer(
                self.config.embedding_model,
                device=device
            )
            
            if device == 'cuda':
                # Don't use FP16 for small GPUs - it can actually use more memory
                gpu_props = torch.cuda.get_device_properties(0)
                gpu_memory_gb = gpu_props.total_memory / 1e9
                
                logger.info(f"✅ GPU acceleration enabled: {torch.cuda.get_device_name(0)}")
                logger.info(f"   GPU memory: {gpu_memory_gb:.2f} GB")
                logger.info(f"   Model dimension: {self.config.embedding_dimension}")
                logger.info(f"   Max sequence length: {self.embedding_model.max_seq_length}")
                
                # Store max tokens for chunking
                self.max_tokens = self.embedding_model.max_seq_length
                
                # Clear cache to start fresh
                torch.cuda.empty_cache()
            else:
                logger.warning("⚠️  Running on CPU - embedding generation will be slower")
                self.max_tokens = self.embedding_model.max_seq_length
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error initializing embeddings: {e}")
            return False
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk text based on model's max token limit.
        Uses the model's tokenizer to respect token boundaries.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks that fit within model's token limit
        """
        # Use 90% of max tokens to leave room for special tokens
        max_chunk_tokens = int(self.max_tokens * 0.9)
        
        # Tokenize the text
        tokens = self.embedding_model.tokenizer.encode(text, add_special_tokens=False)
        
        # If text fits, return as is
        if len(tokens) <= max_chunk_tokens:
            return [text]
        
        # Otherwise, chunk it
        chunks = []
        overlap_tokens = int(max_chunk_tokens * 0.1)  # 10% overlap
        
        for i in range(0, len(tokens), max_chunk_tokens - overlap_tokens):
            chunk_tokens = tokens[i:i + max_chunk_tokens]
            chunk_text = self.embedding_model.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
        
        logger.info(f"Text split into {len(chunks)} chunks (original: {len(tokens)} tokens)")
        return chunks
    
    def process_document(self, doc: Dict[str, Any], collection: str, 
                        colombia_only: bool = False) -> Optional[Dict[str, Any]]:
        """
        Process a single document from any collection.
        
        Args:
            doc: Document from MongoDB
            collection: Collection name (works, person, affiliations)
            colombia_only: If True, only process Colombian entities
            
        Returns:
            Processed document ready for indexing or None if invalid
        """
        try:
            # Import functions based on collection
            if collection == 'works':
                from .text_processing import (
                    extract_work_text, create_work_metadata, 
                    should_index_work, is_colombian_work
                )
                
                # Filter by Colombia if requested
                if colombia_only and not is_colombian_work(doc):
                    return None
                    
                if not should_index_work(doc):
                    return None
                text = extract_work_text(doc)
                metadata = create_work_metadata(doc)
                
            elif collection == 'person':
                from .text_processing import (
                    extract_person_text, create_person_metadata, 
                    should_index_person, is_colombian_person
                )
                
                # Filter by Colombia if requested
                if colombia_only and not is_colombian_person(doc):
                    return None
                    
                if not should_index_person(doc):
                    return None
                text = extract_person_text(doc)
                metadata = create_person_metadata(doc)
                
            elif collection == 'affiliations':
                from .text_processing import (
                    extract_affiliation_text, create_affiliation_metadata, 
                    should_index_affiliation, is_colombian_affiliation
                )
                
                # Filter by Colombia if requested
                if colombia_only and not is_colombian_affiliation(doc):
                    return None
                    
                if not should_index_affiliation(doc):
                    return None
                text = extract_affiliation_text(doc)
                metadata = create_affiliation_metadata(doc)
            else:
                logger.error(f"Unknown collection: {collection}")
                return None
            
            # Chunk text if needed to respect model's token limit
            chunks = self.chunk_text(text)
            
            return {
                'text': chunks,  # Returns list of chunks
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error processing document {doc.get('_id')}: {e}")
            return None
    
    def process_work(self, work: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single work document.
        DEPRECATED: Use process_document instead.
        
        Args:
            work: Work document from MongoDB
            
        Returns:
            Processed document ready for indexing or None if invalid
        """
        return self.process_document(work, 'works')
        try:
            # Check if work should be indexed
            if not should_index_work(work):
                return None
            
            # Extract text and metadata
            text = extract_work_text(work)
            metadata = create_work_metadata(work)
            
            # Chunk text if needed to respect model's token limit
            chunks = self.chunk_text(text)
            
            return {
                'text': chunks,  # Now returns list of chunks
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error processing work {work.get('_id')}: {e}")
            return None
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        Process in sub-batches to avoid GPU OOM.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        try:
            all_embeddings = []
            
            # Process in sub-batches to avoid GPU OOM
            for i in range(0, len(texts), self.config.embedding_batch_size):
                sub_batch = texts[i:i + self.config.embedding_batch_size]
                
                embeddings = self.embedding_model.encode(
                    sub_batch,
                    batch_size=self.config.embedding_batch_size,
                    convert_to_tensor=True,
                    show_progress_bar=False
                )
                
                # Convert to list
                if torch.is_tensor(embeddings):
                    embeddings = embeddings.cpu().numpy()
                
                all_embeddings.extend(embeddings.tolist())
                
                # Clear cache after each sub-batch to prevent memory buildup
                if self.config.use_gpu and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return []
    
    def index_works(self, 
                   collection_name: str = 'works',
                   limit: Optional[int] = None,
                   delete_index: bool = True,
                   colombia_only: bool = False,
                   index_suffix: str = '') -> Dict[str, int]:
        """
        Index documents from MongoDB to OpenSearch.
        Supports works, person, and affiliations collections.
        
        Args:
            collection_name: MongoDB collection name (works, person, affiliations)
            limit: Maximum number of documents to index (None for all)
            delete_index: Whether to delete existing index before indexing
            colombia_only: If True, only index Colombian entities
            index_suffix: Optional suffix for index name (e.g., '_colombia')
            
        Returns:
            Dictionary with statistics
        """
        logger.info("=" * 80)
        logger.info(f"Starting {collection_name} indexing")
        logger.info(f"Collection: {collection_name}")
        logger.info(f"Colombia only: {colombia_only}")
        logger.info(f"Limit: {limit if limit else 'All documents'}")
        logger.info(f"Workers: {self.config.workers}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info("=" * 80)
        
        # Get index name
        index_name = self.config.get_index_name(collection_name) + index_suffix
        
        # Create or recreate index
        if delete_index:
            logger.info(f"Creating index: {index_name}")
            if not self.opensearch_manager.create_index(index_name, delete_if_exists=True):
                logger.error("Failed to create index")
                return {'indexed': 0, 'skipped': 0, 'errors': 0}
        
        # Get collection
        db = self.mongo_client[self.config.mongodb_database]
        collection = db[collection_name]
        
        # Count documents
        query = {}
        total_docs = collection.count_documents(query)
        if limit:
            total_docs = min(total_docs, limit)
        
        logger.info(f"Total documents to process: {total_docs:,}")
        
        # Statistics
        indexed = 0
        skipped = 0
        errors = 0
        start_time = time.time()
        last_update_time = start_time
        last_indexed_count = 0
        
        # Process in batches with parallelization
        cursor = collection.find(query).limit(limit) if limit else collection.find(query)
        batch_docs = []
        
        for doc in cursor:
            batch_docs.append(doc)
            
            # Process batch when full
            if len(batch_docs) >= self.config.batch_size:
                batch_stats = self._process_batch(
                    batch_docs,
                    index_name,
                    collection_name,
                    colombia_only  # Pass filter flag
                )
                
                indexed += batch_stats['indexed']
                skipped += batch_stats['skipped']
                errors += batch_stats['errors']
                
                # Log progress
                current_time = time.time()
                elapsed = current_time - start_time
                docs_since_last = indexed - last_indexed_count
                time_since_last = current_time - last_update_time
                
                current_rate = docs_since_last / time_since_last if time_since_last > 0 else 0
                overall_rate = indexed / elapsed if elapsed > 0 else 0
                progress_pct = (indexed / total_docs * 100) if total_docs > 0 else 0
                
                remaining = total_docs - indexed
                eta_seconds = remaining / overall_rate if overall_rate > 0 else 0
                eta_minutes = eta_seconds / 60
                
                # Progress bar
                bar_length = 40
                filled_length = int(bar_length * indexed // total_docs) if total_docs > 0 else 0
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                
                logger.info(
                    f"[{bar}] {progress_pct:.1f}% | "
                    f"{indexed:,}/{total_docs:,} docs | "
                    f"Speed: {current_rate:.1f} docs/s (avg: {overall_rate:.1f}) | "
                    f"ETA: {eta_minutes:.1f}min | "
                    f"Skipped: {skipped} | Errors: {errors}"
                )
                
                last_update_time = current_time
                last_indexed_count = indexed
                
                # Clear batch
                batch_docs = []
        
        # Process final batch
        if batch_docs:
            batch_stats = self._process_batch(batch_docs, index_name, collection_name, colombia_only)
            indexed += batch_stats['indexed']
            skipped += batch_stats['skipped']
            errors += batch_stats['errors']
        
        # Final statistics
        total_time = time.time() - start_time
        avg_rate = indexed / total_time if total_time > 0 else 0
        
        logger.info("=" * 80)
        logger.info("✅ Indexing completed!")
        logger.info(f"  - Total indexed: {indexed:,} documents")
        logger.info(f"  - Skipped: {skipped:,} documents")
        logger.info(f"  - Errors: {errors:,} documents")
        logger.info(f"  - Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
        logger.info(f"  - Average speed: {avg_rate:.1f} docs/s")
        logger.info("=" * 80)
        
        return {
            'indexed': indexed,
            'skipped': skipped,
            'errors': errors,
            'time': total_time
        }
    
    def _process_batch(self, docs: List[Dict[str, Any]], index_name: str, 
                      collection_name: str, colombia_only: bool = False) -> Dict[str, int]:
        """
        Process a batch of documents with parallel processing.
        
        Args:
            docs: List of documents from MongoDB
            index_name: OpenSearch index name
            collection_name: Collection name (works, person, affiliations)
            colombia_only: If True, only process Colombian entities
            
        Returns:
            Dictionary with batch statistics
        """
        batch_texts = []
        batch_metadatas = []
        skipped = 0
        
        # Process documents in parallel
        with ThreadPoolExecutor(max_workers=self.config.workers) as executor:
            futures = [
                executor.submit(self.process_document, doc, collection_name, colombia_only) 
                for doc in docs
            ]
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    # Result now contains list of chunks
                    text_chunks = result['text']
                    metadata = result['metadata']
                    
                    # Create separate entries for each chunk
                    for chunk_idx, chunk_text in enumerate(text_chunks):
                        batch_texts.append(chunk_text)
                        # Add chunk info to metadata
                        chunk_metadata = metadata.copy()
                        chunk_metadata['chunk_index'] = chunk_idx
                        chunk_metadata['total_chunks'] = len(text_chunks)
                        batch_metadatas.append(chunk_metadata)
                else:
                    skipped += 1
        
        # Generate embeddings
        if not batch_texts:
            return {'indexed': 0, 'skipped': skipped, 'errors': 0}
        
        embeddings = self.generate_embeddings(batch_texts)
        
        if not embeddings:
            return {'indexed': 0, 'skipped': skipped, 'errors': len(batch_texts)}
        
        # Create documents for indexing
        documents = []
        for text, metadata, embedding in zip(batch_texts, batch_metadatas, embeddings):
            # Create unique ID for chunks - detect ID field
            doc_id = metadata.get('work_id') or metadata.get('person_id') or metadata.get('affiliation_id') or ''
            if metadata.get('total_chunks', 1) > 1:
                doc_id = f"{doc_id}_chunk_{metadata['chunk_index']}"
            
            doc = {
                'id': doc_id,
                'text': text,
                'embedding': embedding,
                **metadata
            }
            documents.append(doc)
        
        # Bulk index
        success, errors = self.opensearch_manager.bulk_index(index_name, documents)
        
        return {
            'indexed': success,
            'skipped': skipped,
            'errors': errors
        }
    
    def run(self, 
            collection: str = 'works',
            limit: Optional[int] = None,
            delete_index: bool = True,
            colombia_only: bool = False,
            index_suffix: str = '') -> bool:
        """
        Execute the complete indexing process.
        
        Args:
            collection: Collection name to index
            limit: Maximum number of documents (None for all)
            delete_index: Whether to delete existing index
            colombia_only: If True, only index Colombian entities
            index_suffix: Optional suffix for index name (e.g., '_colombia')
            
        Returns:
            True if successful
        """
        # Connect to services
        if not self.connect_opensearch():
            return False
        
        if not self.connect_mongodb():
            return False
        
        # Initialize embeddings
        if not self.initialize_embeddings():
            return False
        
        # Index documents
        stats = self.index_works(
            collection_name=collection,
            limit=limit,
            delete_index=delete_index,
            colombia_only=colombia_only,
            index_suffix=index_suffix
        )
        
        return stats['indexed'] > 0
