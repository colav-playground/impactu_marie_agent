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
            # Use Prompt Engineer to build search query
            from marie_agent.agents.prompt_engineer import get_prompt_engineer
            from marie_agent.adapters.llm_factory import get_llm_adapter
            
            prompt_engineer = get_prompt_engineer()
            llm = get_llm_adapter()
            
            # Get entities if available
            entities = state.get("entities_resolved", {})
            
            context = {
                "query": query,
                "entities": entities,
                "documents": []
            }
            
            # Build structured query prompt
            query_prompt = prompt_engineer.build_prompt(
                agent_name="retrieval",
                task_description="Generate search terms and filters for OpenSearch",
                context=context,
                technique="structured"  # Use structured for query format
            )
            
            # Get optimized search terms from LLM
            search_terms = llm.generate(query_prompt, max_tokens=100)
            
            # 1. Search OpenSearch for relevant documents
            opensearch_results = self._search_opensearch(search_terms or query)
            
            # 2. Query MongoDB for structured data
            # mongodb_results = self._query_mongodb(query)
            
            # 3. Create evidence items
            evidence_items = self._create_evidence(opensearch_results)
            
            # 4. Update state
            state["retrieved_data"] = opensearch_results
            state["evidence_map"]["opensearch_results"] = evidence_items
            state["evidence_map"]["evidence"] = evidence_items  # Also store in standard location
            
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
    
    def _search_opensearch(self, query: str, limit: int = 10, retry_count: int = 0) -> List[Dict[str, Any]]:
        """
        Search OpenSearch for relevant documents with retry logic.
        
        Args:
            query: User query
            limit: Maximum number of results
            retry_count: Current retry attempt
            
        Returns:
            List of search results
        """
        max_retries = 3
        
        try:
            # Use tool from tools.py for consistency
            from marie_agent.tools import search_publications
            
            logger.info(f"Searching OpenSearch (attempt {retry_count + 1}/{max_retries + 1})")
            
            # Invoke tool - with content_and_artifact, invoke() returns just the formatted string
            # The artifacts are available through tool_call pattern
            result = search_publications.invoke({
                "query": query,
                "limit": limit
            })
            
            # Result is the formatted string, parse documents from OpenSearch directly as fallback
            if isinstance(result, str) and "No publications found" not in result:
                # Success - create documents from the response
                # For now, use direct OpenSearch as this is more reliable
                from opensearchpy import OpenSearch
                
                client = OpenSearch(hosts=[config.opensearch.url], timeout=30)
                index_pattern = f"{config.opensearch.index_prefix}_*"
                search_body = {
                    "query": {"multi_match": {"query": query, "fields": ["text", "title", "authors"]}},
                    "size": limit
                }
                response = client.search(index=index_pattern, body=search_body)
                
                # Convert to expected format
                results = []
                for hit in response["hits"]["hits"]:
                    results.append({
                        "index": "opensearch",
                        "id": hit["_id"],
                        "score": hit["_score"],
                        "source": hit["_source"]
                    })
                
                logger.info(f"✓ Retrieved {len(results)} documents via tool")
                return results
            else:
                logger.warning("Tool returned no results")
                return []
            
        except Exception as e:
            logger.error(f"OpenSearch error (attempt {retry_count + 1}): {e}")
            
            # Retry with exponential backoff
            if retry_count < max_retries:
                import time
                wait_time = 2 ** retry_count
                logger.info(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
                return self._search_opensearch(query, limit, retry_count + 1)
            else:
                logger.error("Max retries reached, returning empty")
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
            source = result["source"]
            text = source.get("text", "")
            
            # Parse text to extract structured data
            parsed = self._parse_document_text(text)
            
            #  Store parsed data in evidence
            evidence = Evidence(
                source_type="opensearch",
                source_id=result["id"],
                collection=result["index"],
                reference=f"opensearch:{result['index']}/{result['id']}",
                content=text[:500],  # Truncate for storage
                confidence=float(result["score"]) / 10.0,  # Normalize score
                timestamp=now
            )
            evidence_items.append(evidence)
            
            # Also create a dict version with parsed fields for easy access
            evidence_items[-1].update({
                "title": parsed.get("title", "Unknown"),
                "abstract": parsed.get("abstract", ""),
                "authors": parsed.get("authors", []),
                "year": parsed.get("year", ""),
                "citations": 0  # TODO: Get from source if available
            })
        
        return evidence_items
    
    def _parse_document_text(self, text: str) -> Dict[str, Any]:
        """
        Parse the text field to extract structured information.
        
        Args:
            text: The text content from OpenSearch
            
        Returns:
            Dictionary with parsed fields
        """
        parsed = {
            "title": "",
            "abstract": "",
            "authors": [],
            "year": ""
        }
        
        lines = text.split("\n")
        
        for line in lines:
            line = line.strip()
            
            # Extract title
            if line.startswith("Title:"):
                # Take first variant before |
                title_part = line.replace("Title:", "").strip()
                if "|" in title_part:
                    parsed["title"] = title_part.split("|")[0].strip()
                else:
                    parsed["title"] = title_part
            
            # Extract abstract
            elif line.startswith("Abstract:"):
                abstract = line.replace("Abstract:", "").strip()
                # Remove language tags like [en], [es]
                if abstract.startswith("["):
                    abstract = abstract[abstract.find("]")+1:].strip()
                parsed["abstract"] = abstract
            
            # Extract authors
            elif line.startswith("Authors:"):
                author_str = line.replace("Authors:", "").strip()
                # Simple split by commas (can be improved)
                if "(" in author_str:
                    # Format: "Name (Institution)"
                    author_name = author_str.split("(")[0].strip()
                    parsed["authors"] = [{"full_name": author_name}]
                else:
                    parsed["authors"] = [{"full_name": author_str}]
            
            # Extract year
            elif line.startswith("Year:"):
                parsed["year"] = line.replace("Year:", "").strip()
        
        return parsed


def retrieval_agent_node(state: AgentState) -> AgentState:
    """
    Retrieval agent node function for LangGraph.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with retrieved evidence
    """
    from marie_agent.core.routing import increment_step
    
    agent = RetrievalAgent()
    state = agent.retrieve(state)
    
    # Increment step after completion
    return increment_step(state)

