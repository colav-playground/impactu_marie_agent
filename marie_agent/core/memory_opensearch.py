"""
OpenSearch-based Memory System - Stores and retrieves plans using semantic search.

Uses OpenSearch for:
- Semantic similarity search with embeddings
- Scalable storage
- Better retrieval than keyword matching
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class OpenSearchMemory:
    """
    Base class for OpenSearch-based memory storage.
    
    Features:
    - Semantic search with embeddings
    - K-NN similarity search
    - Scalable storage
    """
    
    def __init__(
        self,
        opensearch_url: str = "http://localhost:9200",
        index_name: str = "impactu_marie_agent_memory",
        embedding_model: str = "BAAI/bge-m3"
    ):
        """
        Initialize OpenSearch memory.
        
        Args:
            opensearch_url: OpenSearch endpoint
            index_name: Index name for this memory type
            embedding_model: Model for generating embeddings
        """
        self.client = OpenSearch(
            hosts=[opensearch_url],
            http_compress=True,
            use_ssl=False,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False
        )
        
        self.index_name = index_name
        
        # Load embedding model on CPU to avoid GPU OOM
        import torch
        device = "cpu"  # Force CPU for memory operations
        self.embedding_model = SentenceTransformer(embedding_model, device=device)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Create index if not exists
        self._create_index_if_not_exists()
        
        logger.info(f"OpenSearch memory initialized: {index_name} (device={device})")
    
    def _create_index_if_not_exists(self) -> None:
        """Create index with K-NN configuration if it doesn't exist."""
        if self.client.indices.exists(index=self.index_name):
            logger.info(f"Index {self.index_name} already exists")
            return
        
        index_body = {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 100
                },
                "number_of_shards": 1,
                "number_of_replicas": 0
            },
            "mappings": {
                "properties": {
                    "id": {"type": "keyword"},
                    "task": {"type": "text"},
                    "task_embedding": {
                        "type": "knn_vector",
                        "dimension": self.embedding_dim,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                            "parameters": {
                                "ef_construction": 128,
                                "m": 24
                            }
                        }
                    },
                    "content": {"type": "object", "enabled": True},
                    "success": {"type": "boolean"},
                    "metadata": {"type": "object", "enabled": True},
                    "created_at": {"type": "date"},
                    "usage_count": {"type": "integer"}
                }
            }
        }
        
        try:
            self.client.indices.create(index=self.index_name, body=index_body)
            logger.info(f"Created index {self.index_name}")
        except Exception as e:
            logger.error(f"Error creating index: {e}")
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        return self.embedding_model.encode(text, normalize_embeddings=True).tolist()
    
    def search_similar(
        self,
        query: str,
        min_score: float = 0.7,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar entries using K-NN.
        
        Args:
            query: Query text
            min_score: Minimum similarity score (0-1)
            top_k: Number of results to return
            filters: Additional filters (e.g., {"success": True})
            
        Returns:
            List of similar entries with scores
        """
        try:
            query_embedding = self._generate_embedding(query)
            
            # Build query
            knn_query = {
                "size": top_k,
                "query": {
                    "bool": {
                        "must": [
                            {
                                "knn": {
                                    "task_embedding": {
                                        "vector": query_embedding,
                                        "k": top_k
                                    }
                                }
                            }
                        ]
                    }
                }
            }
            
            # Add filters
            if filters:
                for key, value in filters.items():
                    knn_query["query"]["bool"]["must"].append({
                        "term": {key: value}
                    })
            
            # Search
            response = self.client.search(index=self.index_name, body=knn_query)
            
            # Filter by score and format results
            results = []
            for hit in response["hits"]["hits"]:
                score = hit["_score"]
                if score >= min_score:
                    result = hit["_source"]
                    result["similarity_score"] = score
                    results.append(result)
            
            logger.info(f"Found {len(results)} similar entries (min_score={min_score})")
            return results
            
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []


class PlanMemoryOpenSearch(OpenSearchMemory):
    """
    OpenSearch-based plan memory.
    
    Index: impactu_marie_agent_plan_memory
    """
    
    def __init__(
        self,
        opensearch_url: str = "http://localhost:9200",
        embedding_model: str = "BAAI/bge-m3"
    ):
        """Initialize plan memory."""
        super().__init__(
            opensearch_url=opensearch_url,
            index_name="impactu_marie_agent_plan_memory",
            embedding_model=embedding_model
        )
    
    def save_plan(
        self,
        task: str,
        plan_steps: List[Dict[str, Any]],
        success: bool,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a plan to memory.
        
        Args:
            task: Task description
            plan_steps: List of plan steps
            success: Whether execution was successful
            metadata: Additional metadata
            
        Returns:
            Plan ID
        """
        plan_id = f"plan_{datetime.utcnow().timestamp()}"
        
        # Generate embedding
        task_embedding = self._generate_embedding(task)
        
        document = {
            "id": plan_id,
            "task": task,
            "task_embedding": task_embedding,
            "content": {
                "plan_steps": plan_steps
            },
            "success": success,
            "metadata": metadata or {},
            "created_at": datetime.utcnow().isoformat(),
            "usage_count": 0
        }
        
        try:
            self.client.index(
                index=self.index_name,
                id=plan_id,
                body=document
            )
            logger.info(f"Saved plan {plan_id} for task: {task[:50]}...")
            return plan_id
            
        except Exception as e:
            logger.error(f"Error saving plan: {e}")
            return plan_id
    
    def retrieve_similar_plan(
        self,
        task: str,
        min_similarity: float = 0.7
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve most similar successful plan.
        
        Args:
            task: Task description
            min_similarity: Minimum similarity threshold
            
        Returns:
            Most similar plan or None
        """
        results = self.search_similar(
            query=task,
            min_score=min_similarity,
            top_k=1,
            filters={"success": True}
        )
        
        if results:
            plan = results[0]
            
            # Increment usage count
            try:
                self.client.update(
                    index=self.index_name,
                    id=plan["id"],
                    body={
                        "script": {
                            "source": "ctx._source.usage_count += 1",
                            "lang": "painless"
                        }
                    }
                )
            except Exception as e:
                logger.warning(f"Could not update usage count: {e}")
            
            logger.info(f"Retrieved similar plan (similarity={plan['similarity_score']:.2f})")
            return plan
        
        return None
    
    def get_all_plans(self, successful_only: bool = True) -> List[Dict[str, Any]]:
        """Get all stored plans."""
        query = {
            "size": 1000,
            "query": {"match_all": {}}
        }
        
        if successful_only:
            query["query"] = {"term": {"success": True}}
        
        try:
            response = self.client.search(index=self.index_name, body=query)
            return [hit["_source"] for hit in response["hits"]["hits"]]
        except Exception as e:
            logger.error(f"Error getting plans: {e}")
            return []


class EpisodicMemoryOpenSearch(OpenSearchMemory):
    """
    OpenSearch-based episodic memory.
    
    Index: impactu_marie_agent_episodic_memory
    """
    
    def __init__(
        self,
        opensearch_url: str = "http://localhost:9200",
        embedding_model: str = "BAAI/bge-m3"
    ):
        """Initialize episodic memory."""
        super().__init__(
            opensearch_url=opensearch_url,
            index_name="impactu_marie_agent_episodic_memory",
            embedding_model=embedding_model
        )
    
    def save_episode(
        self,
        query: str,
        response: str,
        plan_used: List[Dict[str, Any]],
        success: bool,
        quality_score: Optional[float] = None,
        user_feedback: Optional[str] = None
    ) -> str:
        """
        Save an interaction episode.
        
        Args:
            query: User query
            response: Agent response
            plan_used: Plan that was executed
            success: Whether episode was successful
            quality_score: Quality score if available
            user_feedback: User feedback if provided
            
        Returns:
            Episode ID
        """
        episode_id = f"episode_{datetime.utcnow().timestamp()}"
        
        # Generate embedding from query
        query_embedding = self._generate_embedding(query)
        
        document = {
            "id": episode_id,
            "task": query,  # For K-NN search
            "task_embedding": query_embedding,
            "content": {
                "query": query,
                "response": response,
                "plan_used": plan_used
            },
            "success": success,
            "metadata": {
                "quality_score": quality_score,
                "user_feedback": user_feedback
            },
            "created_at": datetime.utcnow().isoformat(),
            "usage_count": 0
        }
        
        try:
            self.client.index(
                index=self.index_name,
                id=episode_id,
                body=document
            )
            logger.info(f"Saved episode {episode_id}")
            return episode_id
            
        except Exception as e:
            logger.error(f"Error saving episode: {e}")
            return episode_id
    
    def get_recent_episodes(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get recent episodes."""
        query = {
            "size": n,
            "sort": [{"created_at": {"order": "desc"}}],
            "query": {"match_all": {}}
        }
        
        try:
            response = self.client.search(index=self.index_name, body=query)
            return [hit["_source"] for hit in response["hits"]["hits"]]
        except Exception as e:
            logger.error(f"Error getting episodes: {e}")
            return []
    
    def get_successful_episodes(self) -> List[Dict[str, Any]]:
        """Get all successful episodes."""
        return self.search_similar(
            query="",  # Empty query for all
            min_score=0.0,
            top_k=1000,
            filters={"success": True}
        )


# Global instances
_plan_memory_os: Optional[PlanMemoryOpenSearch] = None
_episodic_memory_os: Optional[EpisodicMemoryOpenSearch] = None


def get_plan_memory_opensearch() -> PlanMemoryOpenSearch:
    """Get global OpenSearch plan memory instance."""
    global _plan_memory_os
    if _plan_memory_os is None:
        _plan_memory_os = PlanMemoryOpenSearch()
    return _plan_memory_os


def get_episodic_memory_opensearch() -> EpisodicMemoryOpenSearch:
    """Get global OpenSearch episodic memory instance."""
    global _episodic_memory_os
    if _episodic_memory_os is None:
        _episodic_memory_os = EpisodicMemoryOpenSearch()
    return _episodic_memory_os
