"""
Retrieval Agent - Query MongoDB and OpenSearch for evidence.

Responsible for:
- Executing queries against MongoDB collections
- Performing RAG searches in OpenSearch
- Combining structured and unstructured data
- Returning evidence bundles with metadata
"""

from typing import Dict, Any, List
import logging
from datetime import datetime

from pymongo import MongoClient
from opensearchpy import OpenSearch

from marie_agent.state import AgentState, Evidence, add_audit_event
from marie_agent.config import config

logger = logging.getLogger(__name__)


class RetrievalAgent:
    """
    Agent responsible for retrieving evidence from data sources.
    
    Queries both MongoDB (structured data) and OpenSearch (RAG).
    """
    
    def __init__(self):
        """Initialize retrieval agent with database connections."""
        # MongoDB connection
        self.mongo_client = MongoClient(config.mongodb.uri)
        self.mongo_db = self.mongo_client[config.mongodb.database]
        
        # OpenSearch connection
        self.opensearch = OpenSearch(
            hosts=[config.opensearch.url],
            use_ssl=False,
            verify_certs=False
        )
        
        logger.info("Retrieval agent initialized")
    
    def retrieve(self, state: AgentState) -> AgentState:
        """
        Execute retrieval for the query.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with retrieved data and evidence
        """
        query = state["user_query"]
        logger.info(f"Retrieving evidence for: {query}")
        
        try:
            # TODO: Parse query to extract entities and filters
            # For now, do a simple search
            
            # 1. Search OpenSearch for relevant documents
            opensearch_results = self._search_opensearch(query)
            
            # 2. Query MongoDB for structured data
            # mongodb_results = self._query_mongodb(query)
            
            # 3. Create evidence items
            evidence_items = self._create_evidence(opensearch_results)
            
            # 4. Update state
            state["retrieved_data"] = opensearch_results
            state["evidence_map"]["opensearch_results"] = evidence_items
            
            add_audit_event(state, "retrieval_completed", {
                "opensearch_hits": len(opensearch_results),
                "evidence_items": len(evidence_items)
            })
            
            state["next_agent"] = "validation"
            
            logger.info(f"Retrieved {len(opensearch_results)} documents")
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}", exc_info=True)
            state["error"] = str(e)
            state["next_agent"] = None
            state["status"] = "failed"
        
        return state
    
    def _search_opensearch(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search OpenSearch for relevant documents.
        
        Args:
            query: User query
            limit: Maximum number of results
            
        Returns:
            List of search results
        """
        try:
            # Search across all indices
            index_pattern = f"{config.opensearch.index_prefix}_*"
            
            # Simple text search for now
            # TODO: Add hybrid search with filters
            search_body = {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["text", "title", "authors"],
                        "type": "best_fields"
                    }
                },
                "size": limit
            }
            
            response = self.opensearch.search(
                index=index_pattern,
                body=search_body
            )
            
            results = []
            for hit in response["hits"]["hits"]:
                results.append({
                    "index": hit["_index"],
                    "id": hit["_id"],
                    "score": hit["_score"],
                    "source": hit["_source"]
                })
            
            return results
            
        except Exception as e:
            logger.error(f"OpenSearch query error: {e}")
            return []
    
    def _query_mongodb(self, query: str) -> List[Dict[str, Any]]:
        """
        Query MongoDB for structured data.
        
        Args:
            query: User query
            
        Returns:
            List of MongoDB documents
        """
        # TODO: Parse query and build MongoDB query
        # For now, return empty
        return []
    
    def _create_evidence(self, results: List[Dict[str, Any]]) -> List[Evidence]:
        """
        Convert search results to evidence items.
        
        Args:
            results: Search results
            
        Returns:
            List of evidence items
        """
        evidence_items = []
        now = datetime.utcnow().isoformat()
        
        for result in results:
            evidence = Evidence(
                source_type="opensearch",
                source_id=result["id"],
                collection=result["index"],
                reference=f"opensearch:{result['index']}/{result['id']}",
                content=result["source"].get("text", "")[:500],  # Truncate
                confidence=float(result["score"]) / 10.0,  # Normalize score
                timestamp=now
            )
            evidence_items.append(evidence)
        
        return evidence_items


def retrieval_agent_node(state: AgentState) -> AgentState:
    """
    Retrieval agent node function for LangGraph.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with retrieved evidence
    """
    agent = RetrievalAgent()
    return agent.retrieve(state)
