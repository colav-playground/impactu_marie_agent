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
from dataclasses import dataclass, field
from datetime import datetime

from marie_agent.state import AgentState, add_audit_event
from marie_agent.config import config
from marie_agent.agents.prompt_engineer import get_prompt_engineer
from marie_agent.adapters.llm_factory import get_llm_adapter

logger = logging.getLogger(__name__)


@dataclass
class QueryAttempt:
    """Record of a query attempt with reflection."""
    query: Dict[str, Any]
    result_count: int
    total_hits: int
    reflection: str
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = False


class QueryMemory:
    """Memory system for storing and learning from past queries."""
    
    def __init__(self, max_size: int = 50):
        self.attempts: List[QueryAttempt] = []
        self.max_size = max_size
        self.successful_patterns: List[Dict] = []
    
    def add_attempt(self, attempt: QueryAttempt):
        """Add a query attempt to memory."""
        self.attempts.append(attempt)
        
        # Keep only recent attempts
        if len(self.attempts) > self.max_size:
            self.attempts = self.attempts[-self.max_size:]
        
        # Learn from successful queries
        if attempt.success and attempt.total_hits > 0:
            pattern = {
                "query_type": self._extract_query_type(attempt.query),
                "fields_used": self._extract_fields(attempt.query),
                "result_count": attempt.total_hits
            }
            self.successful_patterns.append(pattern)
            logger.debug(f"Learned successful pattern: {pattern}")
    
    def _extract_query_type(self, query: Dict) -> str:
        """Extract the type of query."""
        if "query" in query:
            q = query["query"]
            if "multi_match" in q:
                return "multi_match"
            elif "bool" in q:
                return "bool"
            elif "match" in q:
                return "match"
        return "unknown"
    
    def _extract_fields(self, query: Dict) -> List[str]:
        """Extract fields used in query."""
        fields = []
        if "query" in query:
            q = query["query"]
            if "multi_match" in q:
                fields = q["multi_match"].get("fields", [])
            elif "match" in q:
                fields = list(q["match"].keys())
        return fields
    
    def get_insights(self) -> str:
        """Get insights from past attempts."""
        if not self.attempts:
            return "No previous attempts"
        
        recent = self.attempts[-5:]
        insights = []
        
        # Analyze recent failures
        failures = [a for a in recent if not a.success]
        if failures:
            insights.append(f"Recent {len(failures)} attempts had poor results")
            if failures[-1].reflection:
                insights.append(f"Last reflection: {failures[-1].reflection}")
        
        # Share successful patterns
        if self.successful_patterns:
            last_success = self.successful_patterns[-1]
            insights.append(f"Successful pattern: {last_success['query_type']} with fields {last_success['fields_used']}")
        
        return " | ".join(insights) if insights else "No specific insights yet"


class OpenSearchExpertAgent:
    """
    Expert agent for OpenSearch operations with iterative refinement.
    
    Features:
    - Reflexion: Reflects on failed queries and improves them
    - Query Memory: Learns from past attempts
    - Iterative Refinement: Tries up to MAX_ITERATIONS times
    - Self-Evaluation: Evaluates quality of results
    
    Workflow:
    1. INSPECT: Get index mappings and understand data structure
    2. GENERATE: Create query dynamically using LLM
    3. EXECUTE: Run query and evaluate results
    4. REFLECT: If poor results, reflect and regenerate (up to MAX_ITERATIONS)
    5. OPTIMIZE: Learn from successful patterns
    """
    
    MAX_ITERATIONS = 3  # Maximum refinement attempts
    MIN_RESULTS_THRESHOLD = 3  # Minimum results to consider success
    
    def __init__(self):
        """Initialize OpenSearch expert with memory."""
        self.client = OpenSearch(hosts=[config.opensearch.url], timeout=30)
        self.index_prefix = config.opensearch.index_prefix
        self.llm = get_llm_adapter()
        self.prompt_engineer = get_prompt_engineer()
        self.schema_cache = {}
        self.query_memory = QueryMemory()  # Add memory system
        logger.info("OpenSearch Expert agent initialized with reflexion capabilities")
    
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
    
    def reflect_on_query(self, query: Dict[str, Any], user_request: str, result_count: int, 
                        total_hits: int, iteration: int) -> str:
        """
        Reflect on query performance and suggest improvements.
        
        Uses Reflexion pattern to analyze what went wrong and how to improve.
        
        Args:
            query: The query that was executed
            user_request: Original user request
            result_count: Number of results retrieved
            total_hits: Total hits found
            iteration: Current iteration number
            
        Returns:
            Reflection text with improvement suggestions
        """
        try:
            logger.info(f"🤔 Reflecting on query (iteration {iteration})...")
            
            # Get insights from memory
            memory_insights = self.query_memory.get_insights()
            
            reflection_prompt = f"""You are an OpenSearch expert reflecting on a query that didn't work well.

USER REQUEST: "{user_request}"

QUERY USED:
{json.dumps(query, indent=2)}

RESULTS:
- Documents retrieved: {result_count}
- Total hits: {total_hits}
- Success: {"No" if total_hits < self.MIN_RESULTS_THRESHOLD else "Yes"}

PAST ATTEMPTS INSIGHTS: {memory_insights}

TASK: Reflect on why this query may have failed and suggest specific improvements.

Consider:
1. Are the fields appropriate for the search?
2. Is the query text too specific or too broad?
3. Should we use different field combinations?
4. What can we learn from past successful patterns?

Provide a concise reflection (2-3 sentences) with actionable improvements:"""

            reflection = self.llm.generate(reflection_prompt, max_tokens=200)
            
            logger.info(f"💭 Reflection: {reflection[:100]}...")
            
            return reflection.strip()
            
        except Exception as e:
            logger.error(f"Error in reflection: {e}")
            return "Try using broader terms and different field combinations"
    
    def evaluate_results(self, result_count: int, total_hits: int) -> bool:
        """
        Evaluate if query results are satisfactory.
        
        Args:
            result_count: Number of results retrieved
            total_hits: Total hits available
            
        Returns:
            True if results are satisfactory
        """
        is_success = total_hits >= self.MIN_RESULTS_THRESHOLD
        
        if is_success:
            logger.info(f"✅ Query evaluation: SUCCESS ({total_hits} hits)")
        else:
            logger.warning(f"❌ Query evaluation: POOR ({total_hits} hits, need >= {self.MIN_RESULTS_THRESHOLD})")
        
        return is_success
    
    def generate_query_dynamically(self, user_request: str, index_structure: Dict[str, Any],
                                  reflection: Optional[str] = None, iteration: int = 1) -> Dict[str, Any]:
        """
        Generate OpenSearch query dynamically based on request and schema.
        
        Uses LLM to understand the request and create appropriate query.
        Incorporates reflection from previous attempts if available.
        
        Args:
            user_request: What the user wants to search
            index_structure: Schema information from inspect_index_structure
            reflection: Optional reflection from previous failed attempt
            iteration: Current iteration number (for logging)
            
        Returns:
            OpenSearch query dictionary
        """
        try:
            logger.info(f"🤖 Generating query (iteration {iteration})" + 
                       (" with reflection" if reflection else ""))
            
            # Prepare schema info for LLM
            available_fields = index_structure.get("common_fields", [])
            
            # Create prompt for query generation - KEEP IT SIMPLE
            text_fields = [f for f in available_fields if f in ['title', 'text', 'abstract', 'keywords', 'description', 'authors']]
            if not text_fields:
                text_fields = ['title', 'text']  # Fallback
            
            # Add reflection to prompt if available
            reflection_guidance = ""
            if reflection:
                reflection_guidance = f"\n\nPREVIOUS ATTEMPT FAILED. REFLECTION:\n{reflection}\n\nIMPROVE the query based on this reflection."
            
            query_gen_prompt = f"""Generate a SIMPLE OpenSearch query JSON.

USER WANTS TO SEARCH FOR: "{user_request}"

USE THESE TEXT FIELDS: {text_fields[:4]}

RULE: Generate ONLY multi_match query - no filters, no bool, no ranges.{reflection_guidance}

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
        Execute OpenSearch query with iterative refinement using Reflexion.
        
        Workflow:
        1. Inspect index structure
        2. Generate initial query
        3. Execute and evaluate results
        4. If poor results: Reflect → Improve → Retry (up to MAX_ITERATIONS)
        5. Return best results found
        
        Args:
            state: Current agent state with query requirements
            
        Returns:
            Updated state with search results and optimization metadata
        """
        query = state.get("user_query", "")
        search_context = state.get("search_context", {})
        requested_size = search_context.get("limit", 10)
        
        logger.info(f"🔍 OpenSearch Expert with Reflexion: {query[:50]}...")
        
        try:
            # STEP 1: Inspect index structure
            index_pattern = f"{self.index_prefix}_*"
            logger.info(f"Step 1: Inspecting index structure: {index_pattern}")
            
            structure = self.inspect_index_structure(index_pattern)
            
            if "error" in structure:
                logger.error(f"Failed to inspect structure: {structure['error']}")
                state["error"] = f"Index inspection failed: {structure['error']}"
                return state
            
            # Initialize iteration variables
            best_query = None
            best_results = []
            best_total = 0
            reflection = None
            
            # ITERATIVE REFINEMENT LOOP
            for iteration in range(1, self.MAX_ITERATIONS + 1):
                logger.info(f"\n{'='*60}")
                logger.info(f"ITERATION {iteration}/{self.MAX_ITERATIONS}")
                logger.info(f"{'='*60}")
                
                # STEP 2: Generate query (with reflection if not first attempt)
                logger.info(f"Step 2.{iteration}: Generating query...")
                
                opensearch_query = self.generate_query_dynamically(
                    query, structure, reflection, iteration
                )
                
                # Add size to query
                opensearch_query["size"] = requested_size
                
                logger.info(f"Generated query: {json.dumps(opensearch_query, indent=2)[:300]}...")
                
                # STEP 3: Execute query
                logger.info(f"Step 3.{iteration}: Executing query")
                
                try:
                    response = self.client.search(
                        index=index_pattern,
                        body=opensearch_query
                    )
                    
                    hits = response["hits"]["hits"]
                    total = response["hits"]["total"]["value"]
                    
                    logger.info(f"✓ Results: {len(hits)} docs (total: {total})")
                    
                    # Format results
                    results = []
                    for hit in hits:
                        results.append({
                            "id": hit["_id"],
                            "index": hit["_index"],
                            "score": hit["_score"],
                            "source": hit["_source"]
                        })
                    
                    # STEP 4: Evaluate results
                    is_success = self.evaluate_results(len(results), total)
                    
                    # Record attempt in memory
                    attempt = QueryAttempt(
                        query=opensearch_query,
                        result_count=len(results),
                        total_hits=total,
                        reflection=reflection or "",
                        success=is_success
                    )
                    self.query_memory.add_attempt(attempt)
                    
                    # Keep best results
                    if total > best_total:
                        best_query = opensearch_query
                        best_results = results
                        best_total = total
                        logger.info(f"🏆 New best: {total} hits")
                    
                    # SUCCESS: Exit loop if results are satisfactory
                    if is_success:
                        logger.info(f"🎉 SUCCESS on iteration {iteration}!")
                        break
                    
                    # STEP 5: Reflect and prepare for next iteration
                    if iteration < self.MAX_ITERATIONS:
                        logger.info(f"Step 5.{iteration}: Reflecting on results...")
                        reflection = self.reflect_on_query(
                            opensearch_query, query, len(results), total, iteration
                        )
                        logger.info(f"Will retry with improved query...")
                    else:
                        logger.warning(f"Max iterations reached. Using best results found.")
                
                except Exception as exec_error:
                    logger.error(f"Query execution error: {exec_error}")
                    # Continue to next iteration with reflection
                    if iteration < self.MAX_ITERATIONS:
                        reflection = f"Previous query caused error: {str(exec_error)}. Try simpler query structure."
            
            # FINAL STEP: Update state with best results
            state["opensearch_results"] = best_results
            state["opensearch_query"] = best_query
            state["opensearch_total"] = best_total
            state["opensearch_structure"] = structure
            state["opensearch_iterations"] = iteration
            
            add_audit_event(state, "opensearch_query_executed", {
                "query_generated": "dynamic_with_reflexion",
                "hits": len(best_results),
                "total": best_total,
                "iterations": iteration,
                "memory_size": len(self.query_memory.attempts),
                "success": best_total >= self.MIN_RESULTS_THRESHOLD
            })
            
            logger.info(f"\n✅ OpenSearch Expert completed: {best_total} hits after {iteration} iteration(s)")
            
        except Exception as e:
            logger.error(f"Error in OpenSearch Expert: {e}", exc_info=True)
            state["error"] = str(e)
        
        return state
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


def opensearch_expert_node(state: AgentState) -> AgentState:
    """
    OpenSearch Expert agent node function for LangGraph.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with OpenSearch results
    """
    agent = OpenSearchExpertAgent()
    return agent.execute_query(state)
