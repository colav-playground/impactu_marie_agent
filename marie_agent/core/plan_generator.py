"""
Dynamic Plan Generator - Creates adaptive plans based on query analysis.

Analyzes queries and generates appropriate execution plans with the right
sequence of agents. Uses memory to retrieve similar successful plans.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PlanStep:
    """A step in the execution plan with enhanced metadata for task decomposition."""
    agent_name: str   # Which agent executes this
    title: str        # Short description
    details: str      # Detailed instructions
    metadata: Dict[str, Any] = None  # Enhanced metadata (task_id, dependencies, variables)
    
    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with all metadata."""
        return {
            "agent_name": self.agent_name,
            "title": self.title,
            "details": self.details,
            "metadata": self.metadata
        }


class DynamicPlanGenerator:
    """
    Generates execution plans based on query analysis.
    
    Routes to appropriate agents based on:
    - Query type (conceptual vs data-driven)
    - Complexity
    - Required information
    - Similar past plans (memory)
    """
    
    def __init__(self, llm, use_memory: bool = True):
        """
        Initialize plan generator.
        
        Args:
            llm: LLM for plan generation
            use_memory: Use plan memory for retrieval
        """
        self.llm = llm
        self.use_memory = use_memory
        
        if use_memory:
            try:
                # Try OpenSearch memory first (better)
                from marie_agent.core.memory_opensearch import get_plan_memory_opensearch
                self.plan_memory = get_plan_memory_opensearch()
                logger.info("Plan memory enabled (OpenSearch)")
            except Exception as e:
                logger.warning(f"Could not load OpenSearch memory: {e}")
                try:
                    # Fallback to JSON memory
                    from marie_agent.core.memory import get_plan_memory
                    self.plan_memory = get_plan_memory()
                    logger.info("Plan memory enabled (JSON fallback)")
                except Exception as e2:
                    logger.warning(f"Could not load any plan memory: {e2}")
                    self.plan_memory = None
        else:
            self.plan_memory = None
    
    def generate_plan(
        self,
        query: str,
        context: str = ""
    ) -> List[PlanStep]:
        """
        Generate execution plan for query.
        
        Workflow:
        1. Check memory for similar successful plans
        2. If found, reuse plan
        3. Otherwise, generate new plan
        
        Args:
            query: User query
            context: Additional context
            
        Returns:
            List of plan steps
        """
        logger.info(f"Generating plan for query: {query[:100]}...")
        
        try:
            # Try to retrieve similar plan from memory
            if self.plan_memory:
                similar_plan = self.plan_memory.retrieve_similar_plan(query, min_similarity=0.7)
                
                if similar_plan:
                    logger.info(f"📚 Retrieved similar plan from memory (used {similar_plan.get('usage_count', 0)} times)")
                    # Convert stored plan steps to PlanStep objects
                    # Handle both OpenSearch (content.plan_steps) and JSON (plan_steps) formats
                    plan_steps_data = similar_plan.get("content", {}).get("plan_steps") or similar_plan.get("plan_steps", [])
                    plan_steps = [
                        PlanStep(**step) for step in plan_steps_data
                    ]
                    return plan_steps
            
            # No similar plan found, generate new one
            logger.info("🆕 Generating new plan")
            
            # Analyze query type
            query_type = self._analyze_query_type(query)
            
            # Generate plan based on type
            if query_type == "greeting":
                plan = self._plan_for_greeting(query)
            elif query_type == "conversational":
                plan = self._plan_for_conversational(query)
            elif query_type == "conceptual":
                plan = self._plan_for_conceptual_query(query)
            elif query_type == "data_driven":
                plan = self._plan_for_data_query(query)
            elif query_type == "complex":
                plan = self._plan_for_complex_query(query)
            else:
                plan = self._plan_default(query)
            
            logger.info(f"Generated plan with {len(plan)} steps")
            return plan
            
        except Exception as e:
            logger.error(f"Error generating plan: {e}", exc_info=True)
            # Fallback: simple retrieval + reporting
            return [
                PlanStep(
                    agent_name="retrieval",
                    title="Retrieve information",
                    details=f"Search for: {query}"
                ),
                PlanStep(
                    agent_name="reporting",
                    title="Generate response",
                    details=f"Answer query: {query}"
                )
            ]
    
    def refine_plan(
        self,
        original_plan: List[PlanStep],
        issues: List[str],
        suggestions: List[str],
        context: str
    ) -> List[PlanStep]:
        """
        Refine plan based on quality feedback.
        
        Args:
            original_plan: Original execution plan
            issues: Issues found with response
            suggestions: Suggestions for improvement
            context: Current context
            
        Returns:
            Refined plan
        """
        logger.info("Refining plan based on quality feedback")
        
        try:
            prompt = self._build_refinement_prompt(
                original_plan, issues, suggestions, context
            )
            
            response = self.llm.generate(prompt, max_tokens=1024)
            refined_plan = self._parse_plan_from_response(response)
            
            logger.info(f"Refined plan to {len(refined_plan)} steps")
            return refined_plan
            
        except Exception as e:
            logger.error(f"Error refining plan: {e}", exc_info=True)
            # Return original plan if refinement fails
            return original_plan
    
    def _analyze_query_type(self, query: str) -> str:
        """
        Classify query using few-shot prompting for fast, accurate intent detection.
        
        Returns:
            Query type: greeting, conversational, conceptual, data_driven, or complex
        """
        # Few-shot examples for classification
        prompt = f"""Classify query:

"hello" → greeting
"what is AI?" → conceptual
"how many papers?" → data_driven
"thanks" → conversational
"top 10 papers from MIT on AI with h-index" → complex

"{query}" →"""

        try:
            response = self.llm.generate(prompt, max_tokens=5).strip().lower()
            
            # Extract category word
            response = response.split()[0] if response else ""
            
            # Validate and extract
            valid_types = ["greeting", "conversational", "conceptual", "data_driven", "complex"]
            
            # Map Spanish variations to English
            spanish_map = {
                "saludo": "greeting",
                "conversacional": "conversational",
                "conceptual": "conceptual",
                "datos": "data_driven",
                "complejo": "complex",
                "compleja": "complex"
            }
            
            # Check Spanish mapping first
            if response in spanish_map:
                result = spanish_map[response]
                logger.info(f"Query classified as: {result} (from {response})")
                return result
            
            # Direct match
            if response in valid_types:
                logger.info(f"Query classified as: {response}")
                return response
            
            # Partial match (handle variations like "data_driven:" or "category: greeting")
            for valid_type in valid_types:
                if valid_type.replace("_", " ") in response or valid_type.replace("_", "-") in response:
                    logger.info(f"Query classified as: {valid_type}")
                    return valid_type
            
            # Default to data_driven as it's the most common for research queries
            logger.warning(f"Unclear classification ('{response}'), defaulting to data_driven")
            return "data_driven"
            
        except Exception as e:
            logger.error(f"Error analyzing query type: {e}")
            return "data_driven"
    
    def _plan_for_greeting(self, query: str) -> List[PlanStep]:
        """Generate plan for greetings using LLM."""
        return self._generate_plan_with_llm(query, "greeting")
    
    def _plan_for_conversational(self, query: str) -> List[PlanStep]:
        """Generate plan for conversational responses using LLM."""
        return self._generate_plan_with_llm(query, "conversational")
    
    def _plan_for_conceptual_query(self, query: str) -> List[PlanStep]:
        """Generate plan for conceptual queries using LLM."""
        return self._generate_plan_with_llm(query, "conceptual")
    
    def _plan_for_data_query(self, query: str) -> List[PlanStep]:
        """Generate plan for data-driven queries using LLM."""
        return self._generate_plan_with_llm(query, "data_driven")
    
    def _plan_for_complex_query(self, query: str) -> List[PlanStep]:
        """Generate plan for complex queries using LLM."""
        return self._generate_plan_with_llm(query, "complex")
    
    def _generate_plan_with_llm(self, query: str, query_type: str) -> List[PlanStep]:
        """
        Generate execution plan using LLM intelligence with enhanced decomposition.
        
        Args:
            query: User query
            query_type: Type of query (greeting, conversational, conceptual, data_driven, complex)
            
        Returns:
            List of plan steps with task IDs, output variables, and dependencies
        """
        # Simplified prompts for simple queries
        if query_type in ["greeting", "conversational"]:
            prompt = f"""Generate JSON plan for query: "{query}"
Type: {query_type}

Response format:
[{{"task_id":"E1","agent_name":"reporting","task_type":"report","title":"Respond to user","details":"Reply appropriately to: {query}","output_var":"answer","dependencies":[]}}]"""
            max_tokens = 150
        
        elif query_type == "conceptual":
            prompt = f"""Generate JSON plan for conceptual query: "{query}"

Response format:
[{{"task_id":"E1","agent_name":"reporting","task_type":"explain","title":"Explain concept","details":"Provide conceptual explanation for: {query}","output_var":"answer","dependencies":[]}}]"""
            max_tokens = 200
            
        else:  # data_driven or complex - ENHANCED DECOMPOSITION
            prompt = f"""You are an expert query planner. DECOMPOSE this query into atomic sub-tasks:

Query: "{query}"

Analysis:
1. Main intent? (search, comparison, ranking, aggregation)
2. Entities? (researchers, institutions, papers)
3. Metrics? (h-index, citations, count)
4. Filters? (country, year, field)

Valid agents:
- entity_resolution: Resolve institutions/researchers
- retrieval: Search documents/data
- metrics: Calculate h-index, citations
- citations: Format citations
- reporting: Generate answer

Task types: search, resolve, compute, filter, rank, aggregate, report

Output JSON array with:
- task_id: E1, E2, E3...
- agent_name
- task_type  
- title: Short description
- details: Full instructions (use $E1, $E2 for variables)
- output_var: Variable name to store result
- dependencies: [task_ids this depends on]
- inputs: ["$var1", "$var2"] (optional)
- filters: {{}} (optional)

EXAMPLES:

Query: "Top 5 Colombian AI researchers by h-index"
[
  {{"task_id":"E1","agent_name":"retrieval","task_type":"search","title":"Find AI researchers","details":"Search researchers (field=AI, country=Colombia)","output_var":"researchers","dependencies":[],"filters":{{"field":"AI","country":"Colombia"}}}},
  {{"task_id":"E2","agent_name":"metrics","task_type":"compute","title":"Calculate h-index","details":"Compute h-index for $researchers","output_var":"h_indices","dependencies":["E1"],"inputs":["$researchers"]}},
  {{"task_id":"E3","agent_name":"reporting","task_type":"rank","title":"Rank top 5","details":"Rank by $h_indices, take top 5, generate report","output_var":"final_answer","dependencies":["E2"],"inputs":["$h_indices","$researchers"]}}
]

Query: "UdeA ML papers 2020-2024"  
[
  {{"task_id":"E1","agent_name":"entity_resolution","task_type":"resolve","title":"Resolve UdeA","details":"Resolve institution: UdeA","output_var":"institution","dependencies":[]}},
  {{"task_id":"E2","agent_name":"retrieval","task_type":"search","title":"Find ML papers","details":"Search papers (institution=$institution, field=ML, year>=2020)","output_var":"papers","dependencies":["E1"],"inputs":["$institution"],"filters":{{"field":"ML","year_min":2020}}}},
  {{"task_id":"E3","agent_name":"reporting","task_type":"report","title":"Generate report","details":"Report on $papers","output_var":"final_answer","dependencies":["E2"],"inputs":["$papers"]}}
]

Now plan for: "{query}"

JSON array:"""
            max_tokens = 1024

        try:
            response = self.llm.generate(prompt, max_tokens=max_tokens).strip()
            
            # Extract JSON from response
            import json
            import re
            
            # Try to find JSON array in response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                plan_data = json.loads(json_str)
                
                # Validate and create enhanced PlanSteps
                plan = []
                valid_agents = ["entity_resolution", "retrieval", "metrics", "citations", "reporting"]
                
                for step in plan_data:
                    if not isinstance(step, dict):
                        continue
                    
                    agent_name = step.get("agent_name", "")
                    if agent_name not in valid_agents:
                        logger.warning(f"Invalid agent name: {agent_name}, skipping step")
                        continue
                    
                    # Extract enhanced fields
                    task_id = step.get("task_id", f"E{len(plan)+1}")
                    task_type = step.get("task_type", "execute")
                    output_var = step.get("output_var", "")
                    dependencies = step.get("dependencies", [])
                    inputs = step.get("inputs", [])
                    filters = step.get("filters", {})
                    
                    # Build enhanced details with metadata
                    details = step.get("details", f"Process query: {query}")
                    if output_var:
                        details += f" → ${output_var}"
                    if dependencies:
                        details += f" (depends on: {', '.join(dependencies)})"
                    
                    plan.append(PlanStep(
                        agent_name=agent_name,
                        title=step.get("title", f"{task_type.title()} step"),
                        details=details
                    ))
                    
                    # Store metadata for later use (Phase 2 will use this)
                    if not hasattr(plan[-1], 'metadata'):
                        plan[-1].metadata = {}
                    plan[-1].metadata.update({
                        'task_id': task_id,
                        'task_type': task_type,
                        'output_var': output_var,
                        'dependencies': dependencies,
                        'inputs': inputs,
                        'filters': filters
                    })
                
                if plan:
                    logger.info(f"Generated enhanced plan with {len(plan)} steps using LLM")
                    return plan
            
            logger.warning(f"Could not parse LLM response as JSON, using fallback")
            
        except Exception as e:
            logger.error(f"Error generating plan with LLM: {e}")
        
        # Fallback to simple plan based on type
        return self._fallback_plan(query, query_type)
    
    def _fallback_plan(self, query: str, query_type: str) -> List[PlanStep]:
        """Fallback plan when LLM fails."""
        if query_type in ["greeting", "conversational"]:
            return [
                PlanStep(
                    agent_name="reporting",
                    title="Respond to user",
                    details=f"Respond appropriately to: {query}"
                )
            ]
        
        # Default: retrieval + reporting
        return [
            PlanStep(
                agent_name="retrieval",
                title="Search for information",
                details=f"Search for: {query}"
            ),
            PlanStep(
                agent_name="reporting",
                title="Generate response",
                details=f"Answer: {query}"
            )
        ]
    
    def _plan_default(self, query: str) -> List[PlanStep]:
        """Default fallback plan."""
        return self._fallback_plan(query, "complex")
    
    def _build_refinement_prompt(
        self,
        original_plan: List[PlanStep],
        issues: List[str],
        suggestions: List[str],
        context: str
    ) -> str:
        """Build prompt for plan refinement."""
        
        prompt_parts = [
            "You are a plan refinement expert. Improve the execution plan based on feedback.",
            "",
            "ORIGINAL PLAN:"
        ]
        
        for i, step in enumerate(original_plan, 1):
            prompt_parts.append(f"{i}. {step.agent_name}: {step.title}")
        
        prompt_parts.extend(["", "ISSUES WITH RESULT:"])
        for issue in issues:
            prompt_parts.append(f"- {issue}")
        
        prompt_parts.extend(["", "SUGGESTIONS:"])
        for suggestion in suggestions:
            prompt_parts.append(f"- {suggestion}")
        
        prompt_parts.extend([
            "",
            f"CONTEXT:\n{context[:500]}...",
            "",
            "Provide IMPROVED PLAN in this format:",
            "STEP 1:",
            "AGENT: [agent_name]",
            "TITLE: [title]",
            "DETAILS: [details]",
            "",
            "STEP 2:",
            "..."
        ])
        
        return "\n".join(prompt_parts)
    
    def _parse_plan_from_response(self, response: str) -> List[PlanStep]:
        """Parse plan steps from LLM response."""
        
        lines = response.split("\n")
        steps = []
        
        current_step = {}
        for line in lines:
            line = line.strip()
            
            if line.startswith("STEP"):
                # Save previous step if exists
                if current_step.get("agent_name"):
                    steps.append(PlanStep(
                        agent_name=current_step["agent_name"],
                        title=current_step.get("title", "Execute"),
                        details=current_step.get("details", "")
                    ))
                current_step = {}
            
            elif "AGENT:" in line.upper():
                current_step["agent_name"] = line.split(":", 1)[1].strip()
            
            elif "TITLE:" in line.upper():
                current_step["title"] = line.split(":", 1)[1].strip()
            
            elif "DETAILS:" in line.upper():
                current_step["details"] = line.split(":", 1)[1].strip()
        
        # Add last step
        if current_step.get("agent_name"):
            steps.append(PlanStep(
                agent_name=current_step["agent_name"],
                title=current_step.get("title", "Execute"),
                details=current_step.get("details", "")
            ))
        
        return steps if steps else self._plan_default("unknown query")
    
    def save_successful_plan(
        self,
        query: str,
        plan_steps: List[PlanStep],
        quality_score: Optional[float] = None,
        execution_time: Optional[float] = None
    ) -> None:
        """
        Save a successful plan to memory for future reuse.
        
        Args:
            query: Original query
            plan_steps: Executed plan steps
            quality_score: Quality score of result
            execution_time: Time taken to execute
        """
        if not self.plan_memory:
            return
        
        try:
            # Convert PlanStep objects to dicts
            plan_dicts = [step.to_dict() for step in plan_steps]
            
            metadata = {}
            if quality_score is not None:
                metadata["quality_score"] = quality_score
            if execution_time is not None:
                metadata["execution_time"] = execution_time
            
            self.plan_memory.save_plan(
                task=query,
                plan_steps=plan_dicts,
                success=True,
                metadata=metadata
            )
            
            logger.info(f"💾 Saved successful plan to memory")
            
        except Exception as e:
            logger.error(f"Error saving plan to memory: {e}")
