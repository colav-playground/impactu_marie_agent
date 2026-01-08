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
    """A step in the execution plan."""
    agent_name: str   # Which agent executes this
    title: str        # Short description
    details: str      # Detailed instructions
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return {
            "agent_name": self.agent_name,
            "title": self.title,
            "details": self.details
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
            
            response = self.llm.invoke(prompt)
            refined_plan = self._parse_plan_from_response(response)
            
            logger.info(f"Refined plan to {len(refined_plan)} steps")
            return refined_plan
            
        except Exception as e:
            logger.error(f"Error refining plan: {e}", exc_info=True)
            # Return original plan if refinement fails
            return original_plan
    
    def _analyze_query_type(self, query: str) -> str:
        """
        Analyze query to determine type.
        
        Returns:
            Query type: greeting, conversational, conceptual, data_driven, or complex
        """
        query_lower = query.lower().strip()
        
        # Greetings and simple interactions
        greeting_keywords = [
            "hola", "hello", "hi", "hey", "buenos días", "good morning",
            "buenas tardes", "good afternoon", "buenas noches", "good evening",
            "cómo estás", "how are you", "qué tal", "what's up"
        ]
        
        # Conversational queries (not requiring RAG)
        conversational_keywords = [
            "gracias", "thanks", "ok", "vale", "entiendo", "i understand",
            "adiós", "bye", "chao", "hasta luego", "see you"
        ]
        
        # Check for greetings
        if any(query_lower.startswith(kw) or query_lower == kw for kw in greeting_keywords):
            return "greeting"
        
        # Check for conversational
        if any(kw in query_lower for kw in conversational_keywords):
            return "conversational"
        
        # Conceptual queries (explanations)
        conceptual_keywords = [
            "qué es", "what is", "explica", "explain",
            "define", "definición", "concepto", "concept",
            "cómo funciona", "how does", "para qué sirve", "what for"
        ]
        
        # Data-driven queries (counts, lists, stats)
        data_keywords = [
            "cuántos", "how many", "lista", "list",
            "papers", "documentos", "investigadores", "researchers",
            "top", "más citados", "most cited", "ranking",
            "estadísticas", "statistics", "análisis", "analysis"
        ]
        
        # Check for conceptual
        if any(kw in query_lower for kw in conceptual_keywords):
            return "conceptual"
        
        # Check for data-driven
        if any(kw in query_lower for kw in data_keywords):
            return "data_driven"
        
        # Default to data-driven if mentions universities/institutions
        if any(word in query_lower for word in ["universidad", "university", "unal", "udea"]):
            return "data_driven"
        
        return "complex"
    
    def _plan_for_greeting(self, query: str) -> List[PlanStep]:
        """Generate plan for greetings and simple hellos."""
        return [
            PlanStep(
                agent_name="reporting",
                title="Respond to greeting",
                details=(
                    f"User said: '{query}'. Respond with a friendly greeting and offer help. "
                    "Explain that you're MARIE, an AI assistant for scientometric queries about research "
                    "publications, authors, and institutions. Ask how you can help them."
                )
            )
        ]
    
    def _plan_for_conversational(self, query: str) -> List[PlanStep]:
        """Generate plan for conversational responses (thanks, ok, bye, etc.)."""
        return [
            PlanStep(
                agent_name="reporting",
                title="Respond conversationally",
                details=(
                    f"User said: '{query}'. Respond appropriately in a brief, friendly manner. "
                    "If they're saying goodbye, wish them well. If thanking, acknowledge it politely."
                )
            )
        ]
    
    def _plan_for_conceptual_query(self, query: str) -> List[PlanStep]:
        """Generate plan for conceptual queries (definitions, explanations)."""
        return [
            PlanStep(
                agent_name="retrieval",
                title="Find relevant papers and definitions",
                details=f"Search for papers and sources about: {query}"
            ),
            PlanStep(
                agent_name="reporting",
                title="Explain concept with references",
                details=(
                    f"Provide a clear explanation of the concept requested in: {query}. "
                    "Use retrieved papers as supporting references and examples."
                )
            )
        ]
    
    def _plan_for_data_query(self, query: str) -> List[PlanStep]:
        """Generate plan for data-driven queries (counts, lists, rankings)."""
        
        # Check if entity resolution needed
        needs_entity_resolution = any(
            word in query.lower() 
            for word in ["universidad", "university", "unal", "udea", "antioquia"]
        )
        
        plan = []
        
        if needs_entity_resolution:
            plan.append(PlanStep(
                agent_name="entity_resolution",
                title="Resolve institution names",
                details="Identify and standardize institution names in the query"
            ))
        
        plan.append(PlanStep(
            agent_name="retrieval",
            title="Retrieve documents",
            details=f"Search for documents matching: {query}"
        ))
        
        # Check if metrics needed
        if any(word in query.lower() for word in ["cuántos", "how many", "top", "ranking", "más"]):
            plan.append(PlanStep(
                agent_name="metrics",
                title="Compute statistics",
                details="Calculate counts, rankings, and statistics from retrieved documents"
            ))
        
        plan.append(PlanStep(
            agent_name="reporting",
            title="Generate data-driven answer",
            details=f"Answer with specific numbers and data: {query}"
        ))
        
        return plan
    
    def _plan_for_complex_query(self, query: str) -> List[PlanStep]:
        """Generate plan for complex queries requiring multiple steps."""
        return [
            PlanStep(
                agent_name="entity_resolution",
                title="Resolve entities",
                details="Identify institutions, researchers, or topics"
            ),
            PlanStep(
                agent_name="retrieval",
                title="Retrieve information",
                details=f"Search comprehensively for: {query}"
            ),
            PlanStep(
                agent_name="metrics",
                title="Analyze data",
                details="Compute relevant statistics and metrics"
            ),
            PlanStep(
                agent_name="reporting",
                title="Generate comprehensive response",
                details=f"Provide complete answer to: {query}"
            )
        ]
    
    def _plan_default(self, query: str) -> List[PlanStep]:
        """Default plan when query type unclear."""
        return [
            PlanStep(
                agent_name="retrieval",
                title="Search for information",
                details=f"Find relevant information for: {query}"
            ),
            PlanStep(
                agent_name="reporting",
                title="Generate response",
                details=f"Answer query: {query}"
            )
        ]
    
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
