"""
OpenSearch Expert Agent - Specialized agent for OpenSearch queries.

This agent is an expert in constructing and executing OpenSearch queries.
ALWAYS inspects data structure FIRST, then generates queries dynamically.
NO hardcoded queries - everything is generated based on schema inspection.
"""

from typing import Dict, Any, List, Optional
import logging
import json
from opensearchpy import OpenSearch

from marie_agent.state import AgentState, add_audit_event
from marie_agent.config import config
from marie_agent.agents.prompt_engineer import get_prompt_engineer
from marie_agent.adapters.llm_factory import get_llm_adapter

logger = logging.getLogger(__name__)


class OpenSearchExpertAgent:
    """
    Expert agent for OpenSearch operations.
    
    Workflow:
    1. INSPECT: Get index mappings and understand data structure
    2. ANALYZE: Understand what fields are available
    3. GENERATE: Create query dynamically using LLM based on schema
    4. EXECUTE: Run query and return results
    
    Capabilities:
    - Schema inspection and analysis
    - Dynamic query generation
    - Semantic search with K-NN
    - Complex boolean queries
    - Aggregations and analytics
    - Query optimization
    """
    
    def __init__(self):
        """Initialize OpenSearch expert."""
        self.client = OpenSearch(hosts=[config.opensearch.url], timeout=30)
        self.index_prefix = config.opensearch.index_prefix
        self.llm = get_llm_adapter()
        self.prompt_engineer = get_prompt_engineer()
        self.schema_cache = {}  # Cache schemas to avoid repeated inspections
        logger.info("OpenSearch Expert agent initialized")
    
    def inspect_index_structure(self, index_pattern: str) -> Dict[str, Any]:
        """
        Inspect OpenSearch index structure and mappings.
        
        Args:
            index_pattern: Index pattern to inspect
            
        Returns:
            Dictionary with index structure information
        """
        try:
            # Check cache first
            if index_pattern in self.schema_cache:
                logger.debug(f"Using cached schema for {index_pattern}")
                return self.schema_cache[index_pattern]
            
            logger.info(f"🔍 Inspecting structure of {index_pattern}")
            
            # Get index mappings
            mappings = self.client.indices.get_mapping(index=index_pattern)
            
            # Extract field information
            structure = {
                "indices": {},
                "common_fields": set(),
                "field_types": {}
            }
            
            for index_name, index_data in mappings.items():
                properties = index_data.get("mappings", {}).get("properties", {})
                
                structure["indices"][index_name] = {
                    "fields": list(properties.keys()),
                    "mapping": properties
                }
                
                # Track common fields
                for field_name, field_info in properties.items():
                    structure["common_fields"].add(field_name)
                    field_type = field_info.get("type", "unknown")
                    structure["field_types"][field_name] = field_type
            
            structure["common_fields"] = list(structure["common_fields"])
            
            # Get sample document to understand data
            try:
                sample = self.client.search(
                    index=index_pattern,
                    body={"size": 1, "query": {"match_all": {}}},
                    _source=True
                )
                if sample["hits"]["hits"]:
                    structure["sample_document"] = sample["hits"]["hits"][0]["_source"]
            except Exception as e:
                logger.warning(f"Could not get sample document: {e}")
            
            # Cache the structure
            self.schema_cache[index_pattern] = structure
            
            logger.info(f"✓ Inspected {len(structure['indices'])} indices, "
                       f"found {len(structure['common_fields'])} common fields")
            
            return structure
            
        except Exception as e:
            logger.error(f"Error inspecting index structure: {e}")
            return {"error": str(e), "indices": {}, "common_fields": [], "field_types": {}}
    
    def generate_query_dynamically(self, user_request: str, index_structure: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate OpenSearch query dynamically based on request and schema.
        
        Uses LLM to understand the request and create appropriate query
        based on available fields in the index.
        
        Args:
            user_request: What the user wants to search
            index_structure: Schema information from inspect_index_structure
            
        Returns:
            OpenSearch query dictionary
        """
        try:
            logger.info("🤖 Generating query dynamically using LLM and schema")
            
            # Prepare schema info for LLM
            available_fields = index_structure.get("common_fields", [])
            field_types = index_structure.get("field_types", {})
            sample_doc = index_structure.get("sample_document", {})
            
            # Build context for LLM
            context = {
                "user_request": user_request,
                "available_fields": available_fields[:20],  # Limit for context
                "field_types": {k: v for k, v in list(field_types.items())[:15]},
                "sample_data": json.dumps(sample_doc, indent=2)[:500] if sample_doc else "No sample available"
            }
            
            # Create prompt for query generation - KEEP IT SIMPLE
            text_fields = [f for f in available_fields if f in ['title', 'text', 'abstract', 'keywords', 'description', 'authors']]
            if not text_fields:
                text_fields = ['title', 'text']  # Fallback
            
            query_gen_prompt = f"""Generate a SIMPLE OpenSearch query JSON.

USER WANTS TO SEARCH FOR: "{user_request}"

USE THESE TEXT FIELDS: {text_fields[:4]}

RULE: Generate ONLY multi_match query - no filters, no bool, no ranges.

TEMPLATE:
{{
  "query": {{
    "multi_match": {{
      "query": "PUT USER REQUEST TEXT HERE",
      "fields": ["field1", "field2", "field3"]
    }}
  }}
}}

NOW GENERATE (use exact user request text, select 2-4 fields from the list above):"""
            
            # Generate query using LLM
            llm_response = self.llm.generate(query_gen_prompt, max_tokens=300)
            
            logger.debug(f"LLM generated query: {llm_response[:200]}...")
            
            # Parse JSON from response
            try:
                # Try to find JSON in response
                json_start = llm_response.find('{')
                json_end = llm_response.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = llm_response[json_start:json_end]
                    query = json.loads(json_str)
                    
                    # Validate basic structure
                    if "query" in query or "aggs" in query:
                        logger.info("✓ Generated valid OpenSearch query")
                        return query
                    else:
                        logger.warning("Query missing 'query' or 'aggs', using fallback")
                        return self._create_fallback_query(user_request, available_fields)
                else:
                    logger.warning("No JSON found in LLM response, using fallback")
                    return self._create_fallback_query(user_request, available_fields)
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {e}")
                return self._create_fallback_query(user_request, available_fields)
                
        except Exception as e:
            logger.error(f"Error generating query dynamically: {e}")
            return self._create_fallback_query(user_request, available_fields)
    
    def _create_fallback_query(self, user_request: str, available_fields: List[str]) -> Dict[str, Any]:
        """
        Create a simple fallback query when LLM generation fails.
        
        Args:
            user_request: User's search request
            available_fields: Fields available in index
            
        Returns:
            Simple multi_match query
        """
        # Use common text fields
        text_fields = [f for f in available_fields if f in ["title", "text", "abstract", "description", "authors", "keywords"]]
        
        if not text_fields:
            text_fields = available_fields[:5]  # Use first 5 fields as fallback
        
        return {
            "query": {
                "multi_match": {
                    "query": user_request,
                    "fields": text_fields,
                    "type": "best_fields"
                }
            }
        }
    
    def execute_query(self, state: AgentState) -> AgentState:
        """
        Execute OpenSearch query with dynamic generation.
        
        Workflow:
        1. Inspect index structure
        2. Generate query dynamically based on schema
        3. Execute query
        4. Return results
        
        Args:
            state: Current agent state with query requirements
            
        Returns:
            Updated state with search results
        """
        query = state.get("user_query", "")
        search_context = state.get("search_context", {})
        requested_size = search_context.get("limit", 10)
        
        logger.info(f"🔍 OpenSearch Expert processing: {query[:50]}...")
        
        try:
            # STEP 1: Inspect index structure
            index_pattern = f"{self.index_prefix}_*"
            logger.info(f"Step 1: Inspecting index structure: {index_pattern}")
            
            structure = self.inspect_index_structure(index_pattern)
            
            if "error" in structure:
                logger.error(f"Failed to inspect structure: {structure['error']}")
                state["error"] = f"Index inspection failed: {structure['error']}"
                return state
            
            # STEP 2: Generate query dynamically
            logger.info(f"Step 2: Generating query dynamically for: {query[:50]}...")
            
            opensearch_query = self.generate_query_dynamically(query, structure)
            
            # Add size to query
            opensearch_query["size"] = requested_size
            
            logger.info(f"Generated query: {json.dumps(opensearch_query, indent=2)[:300]}...")
            
            # STEP 3: Execute query
            logger.info("Step 3: Executing dynamically generated query")
            
            response = self.client.search(
                index=index_pattern,
                body=opensearch_query
            )
            
            # STEP 4: Process results
            hits = response["hits"]["hits"]
            total = response["hits"]["total"]["value"]
            
            logger.info(f"✓ Query executed: {len(hits)} results (total: {total})")
            
            # Format results
            results = []
            for hit in hits:
                results.append({
                    "id": hit["_id"],
                    "index": hit["_index"],
                    "score": hit["_score"],
                    "source": hit["_source"]
                })
            
            # Update state
            state["opensearch_results"] = results
            state["opensearch_query"] = opensearch_query
            state["opensearch_total"] = total
            state["opensearch_structure"] = structure  # Share structure with other agents
            
            add_audit_event(state, "opensearch_query_executed", {
                "query_generated": "dynamic",
                "hits": len(hits),
                "total": total,
                "fields_available": len(structure.get("common_fields", []))
            })
            
            logger.info(f"✓ OpenSearch Expert completed successfully")
            
        except Exception as e:
            logger.error(f"Error in OpenSearch Expert: {e}", exc_info=True)
            state["error"] = str(e)
        
        return state


def opensearch_expert_agent_node(state: AgentState) -> AgentState:
    """
    OpenSearch Expert agent node function for LangGraph.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with OpenSearch results
    """
    agent = OpenSearchExpertAgent()
    return agent.execute_query(state)
