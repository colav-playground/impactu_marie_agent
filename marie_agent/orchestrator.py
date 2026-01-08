"""
Orchestrator Agent - Central control flow manager.

Based on Magentic Orchestration pattern from Microsoft paper.
Implements dynamic planning, execution with progress ledger, and quality-based refinement.
"""

from typing import Dict, Any, List
import logging
from datetime import datetime

from marie_agent.state import AgentState, add_audit_event, Task
from marie_agent.adapters.llm_factory import get_llm_adapter
from marie_agent.human_interaction import get_human_manager
from marie_agent.services.colombian_detector import get_colombian_detector
from marie_agent.config import config

# Import new Magentic components
from marie_agent.core.context_window import ContextWindow
from marie_agent.core.progress_ledger import ProgressTracker
from marie_agent.core.quality_evaluator import QualityEvaluator
from marie_agent.core.plan_generator import DynamicPlanGenerator, PlanStep

logger = logging.getLogger(__name__)


class OrchestratorAgent:
    """
    Central orchestrator managing the multi-agent workflow.
    
    Operating Modes:
    - Planning Mode: Generates adaptive execution plans
    - Execution Mode: Executes plans with progress tracking
    
    Responsibilities:
    - Analyze query and determine if RAG needed
    - Generate dynamic plans based on query type
    - Execute plans step by step with progress ledger
    - Evaluate response quality and refine if needed
    - Manage context window
    - Trigger human interaction when needed
    """
    
    def __init__(self):
        """Initialize orchestrator with Magentic components."""
        self.llm = get_llm_adapter()
        self.human_manager = get_human_manager()
        
        # Initialize Magentic components
        self.context_window = ContextWindow(max_tokens=8000)
        self.progress_tracker = ProgressTracker(self.llm)
        self.quality_evaluator = QualityEvaluator(self.llm, threshold=0.7)
        self.plan_generator = DynamicPlanGenerator(self.llm, use_memory=True)
        
        # Initialize memory and session management
        try:
            from marie_agent.core.memory_opensearch import get_episodic_memory_opensearch
            from marie_agent.core.action_guard import get_action_guard
            
            self.episodic_memory = get_episodic_memory_opensearch()
            self.action_guard = get_action_guard(self.llm, auto_approve=True)  # Auto-approve for now
            
            logger.info("Memory (OpenSearch) and action guard initialized")
        except Exception as e:
            logger.warning(f"Could not initialize advanced features: {e}")
            try:
                # Fallback to JSON memory
                from marie_agent.core.memory import get_episodic_memory
                from marie_agent.core.action_guard import get_action_guard
                
                self.episodic_memory = get_episodic_memory()
                self.action_guard = get_action_guard(self.llm, auto_approve=True)
                
                logger.info("Memory (JSON fallback) and action guard initialized")
            except Exception as e2:
                logger.warning(f"Could not initialize any memory: {e2}")
                self.episodic_memory = None
                self.action_guard = None
        
        self.max_refinement_iterations = 2  # Max times to refine plan
        
        logger.info("🎭 Orchestrator initialized with Magentic architecture")
    
    def _needs_research_data(self, parsed_query: Dict[str, Any]) -> bool:
        """
        Determine if query requires database access (RAG).
        Uses ONLY OpenSearch for Colombian entity detection.
        
        Args:
            parsed_query: Parsed query with intent, entities, etc.
            
        Returns:
            True if RAG pipeline needed, False for direct LLM response
        """
        query_text = parsed_query.get("original_query", "")
        
        # Step 1: Detect Colombian entities from OpenSearch
        detector = get_colombian_detector()
        colombian_context = detector.detect_colombian_context(query_text)
        
        if colombian_context["has_colombian_entities"]:
            logger.info(f"🇨🇴 RAG needed: Colombian entities detected - {colombian_context['detected_entities']}")
            
            # Check if reindexing needed
            reindex_signal = detector.should_trigger_reindex(query_text)
            if reindex_signal["needs_reindex"]:
                logger.warning(f"⚠️ Missing entities in OpenSearch: {reindex_signal['missing_entities']}")
                logger.warning("💡 RAG indexer should index these entities from MongoDB")
            
            return True
        
        # Step 2: Check for specific entity mentions from LLM parse
        entities = parsed_query.get("entities", {})
        if entities.get("institutions") or entities.get("authors") or entities.get("groups"):
            logger.info("RAG needed: Specific entities mentioned (will verify in OpenSearch)")
            return True
        
        # Check intent
        intent = parsed_query.get("intent", "")
        
        # Research-heavy intents
        research_intents = [
            "top_papers",
            "author_productivity", 
            "collaboration_network",
            "institution_ranking",
            "metrics_analysis",
            "citation_analysis",
            "research_trends"
        ]
        
        if any(ri in intent for ri in research_intents):
            logger.info(f"RAG needed: Research intent detected - {intent}")
            return True
        
        # Check for quantitative requests
        query_lower = parsed_query.get("original_query", "").lower()
        quantitative_keywords = [
            "how many papers",
            "top cited",
            "most productive",
            "h-index",
            "impact factor",
            "citation count",
            "compare productivity"
        ]
        
        if any(kw in query_lower for kw in quantitative_keywords):
            logger.info("RAG needed: Quantitative analysis requested")
            return True
        
        # Check for specific artifact requests
        if parsed_query.get("artifact_type") in ["paper", "patent", "dataset", "software"]:
            logger.info("RAG needed: Specific artifact type requested")
            return True
        
        # General knowledge queries - no RAG needed
        general_intents = [
            "explanation",
            "definition",
            "how_to",
            "opinion",
            "recommendation",
            "greeting"
        ]
        
        if any(gi in intent for gi in general_intents):
            logger.info(f"No RAG needed: General knowledge intent - {intent}")
            return False
        
        # Default: try to detect if question is general or research-specific
        # If unsure, prefer direct response for better UX
        logger.info("No clear research indicators - using direct LLM response")
        return False
    
    def plan(self, state: AgentState) -> AgentState:
        """
        Planning Mode - Generate dynamic execution plan with RAG detection.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with plan
        """
        logger.info(f"🎯 PLANNING MODE - Query: {state['user_query'][:100]}...")
        
        try:
            # Add user query to context
            self.context_window.add_message(
                role="user",
                content=state["user_query"]
            )
            
            # Parse query to determine RAG necessity
            parsed_query = self.llm.parse_query(state["user_query"])
            parsed_query["original_query"] = state["user_query"]
            
            # Decide if RAG is needed
            needs_rag = self._needs_research_data(parsed_query)
            state["needs_rag"] = needs_rag
            state["parsed_query"] = parsed_query
            
            if not needs_rag:
                # General knowledge - direct response
                logger.info("💡 General query - direct LLM response (no RAG)")
                plan = [
                    PlanStep(
                        agent_name="reporting",
                        title="Direct answer",
                        details=f"Answer query directly: {state['user_query']}"
                    )
                ]
            else:
                # Research query - generate dynamic plan
                logger.info("🔬 Research query - generating dynamic plan")
                context = self.context_window.get_context_for_agent("orchestrator")
                plan = self.plan_generator.generate_plan(
                    query=state["user_query"],
                    context=context
                )
            
            # Convert plan to state format
            plan_dict = {
                "mode": "direct_response" if not needs_rag else "research_rag",
                "steps": [step.title for step in plan],
                "agents_required": [step.agent_name for step in plan],
                "plan_steps": [step.to_dict() for step in plan]
            }
            
            state["plan"] = plan_dict
            state["current_step"] = 0
            state["status"] = "executing"
            state["next_agent"] = plan[0].agent_name if plan else "reporting"
            
            # Log plan
            logger.info(f"📋 Generated plan with {len(plan)} steps:")
            for i, step in enumerate(plan, 1):
                logger.info(f"  {i}. {step.agent_name}: {step.title}")
            
            add_audit_event(state, "plan_created", {
                "mode": plan_dict["mode"],
                "steps": len(plan),
                "needs_rag": needs_rag
            })
            
        except Exception as e:
            logger.error(f"Error in planning: {e}", exc_info=True)
            # Fallback plan
            state["plan"] = self._create_default_plan()
            state["next_agent"] = "retrieval"
        
        return state
    
    def _create_default_plan(self) -> Dict[str, Any]:
        """Create default fallback plan."""
        return {
            "mode": "research_rag",
            "steps": [
                "Retrieve evidence",
                "Generate report"
            ],
            "agents_required": [
                "retrieval",
                "reporting"
            ],
            "plan_steps": [
                {
                    "agent_name": "retrieval",
                    "title": "Retrieve evidence",
                    "details": "Search for relevant documents"
                },
                {
                    "agent_name": "reporting",
                    "title": "Generate report",
                    "details": "Create response from evidence"
                }
            ]
        }
    
    def execute_with_progress_tracking(self, state: AgentState) -> AgentState:
        """
        Execution Mode - Execute plan with progress ledger tracking.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state after execution step
        """
        logger.info("⚙️ EXECUTION MODE - Tracking progress")
        
        plan = state.get("plan", {})
        plan_steps = plan.get("plan_steps", [])
        current_step_idx = state.get("current_step", 0)
        
        if current_step_idx >= len(plan_steps):
            logger.info("✅ All plan steps completed")
            state["status"] = "completed"
            return state
        
        # Generate progress ledger for current step
        context = self.context_window.get_context_for_agent("orchestrator")
        
        try:
            ledger = self.progress_tracker.generate_ledger(
                query=state["user_query"],
                plan=plan_steps,
                current_step_index=current_step_idx,
                context=context,
                evidence_map=state["evidence_map"]
            )
            
            logger.info(f"📊 Progress Ledger: {ledger}")
            
            # Check if step complete
            if ledger.step_complete.answer:
                logger.info(f"✓ Step {current_step_idx + 1} complete: {ledger.step_complete.reason}")
                state["current_step"] += 1
                
                # Check if all steps done
                if state["current_step"] >= len(plan_steps):
                    state["status"] = "ready_for_quality_check"
                    state["next_agent"] = None
                else:
                    # Move to next step
                    next_step = plan_steps[state["current_step"]]
                    state["next_agent"] = next_step["agent_name"]
            
            # Check if replan needed
            elif ledger.replan.answer:
                logger.warning(f"⚠️ Replan needed: {ledger.replan.reason}")
                state["status"] = "replanning"
                state["replan_reason"] = ledger.replan.reason
                # Will trigger replanning in orchestrator_node
            
            else:
                # Continue with current step
                current_step = plan_steps[current_step_idx]
                state["next_agent"] = ledger.instruction.agent_name
                logger.info(f"→ Executing: {current_step['agent_name']} - {current_step['title']}")
            
            # Add ledger to audit
            add_audit_event(state, "progress_ledger", {
                "step": current_step_idx + 1,
                "complete": ledger.step_complete.answer,
                "replan": ledger.replan.answer,
                "agent": ledger.instruction.agent_name
            })
            
        except Exception as e:
            logger.error(f"Error in progress tracking: {e}", exc_info=True)
            # Continue with plan as-is
            if state["current_step"] < len(plan_steps):
                state["next_agent"] = plan_steps[state["current_step"]]["agent_name"]
        
        return state
    
    def evaluate_quality_and_refine(self, state: AgentState) -> AgentState:
        """
        Evaluate response quality and refine plan if needed.
        
        Args:
            state: Current agent state with final answer
            
        Returns:
            Updated state (may trigger replanning)
        """
        logger.info("🔍 QUALITY EVALUATION - Checking response quality")
        
        if not state.get("final_answer"):
            logger.warning("No final answer to evaluate")
            state["status"] = "completed"
            return state
        
        try:
            # Get evidence for evaluation
            evidence = []
            for ev_list in state["evidence_map"].values():
                evidence.extend(ev_list)
            
            # Evaluate quality
            quality_report = self.quality_evaluator.evaluate_response(
                query=state["user_query"],
                response=state["final_answer"],
                evidence=evidence,
                context=self.context_window.get_progress_summary()
            )
            
            logger.info(f"Quality Score: {quality_report.score:.2f} "
                       f"({'✓ ACCEPTABLE' if quality_report.is_acceptable else '✗ NEEDS IMPROVEMENT'})")
            
            if quality_report.issues:
                logger.warning("Issues found:")
                for issue in quality_report.issues:
                    logger.warning(f"  - {issue}")
            
            state["quality_report"] = {
                "score": quality_report.score,
                "acceptable": quality_report.is_acceptable,
                "issues": quality_report.issues,
                "suggestions": quality_report.suggestions,
                "dimensions": quality_report.dimensions
            }
            
            # Check if refinement needed
            refinement_count = state.get("refinement_count", 0)
            
            if not quality_report.is_acceptable and refinement_count < self.max_refinement_iterations:
                logger.info(f"🔄 Refining plan (iteration {refinement_count + 1}/{self.max_refinement_iterations})")
                
                # Refine plan
                current_plan_steps = state["plan"].get("plan_steps", [])
                current_plan = [PlanStep(**step) for step in current_plan_steps]
                
                refined_plan = self.plan_generator.refine_plan(
                    original_plan=current_plan,
                    issues=quality_report.issues,
                    suggestions=quality_report.suggestions,
                    context=self.context_window.get_progress_summary()
                )
                
                # Update state with refined plan
                state["plan"]["plan_steps"] = [step.to_dict() for step in refined_plan]
                state["plan"]["agents_required"] = [step.agent_name for step in refined_plan]
                state["current_step"] = 0
                state["refinement_count"] = refinement_count + 1
                state["status"] = "executing"
                state["next_agent"] = refined_plan[0].agent_name if refined_plan else "reporting"
                
                logger.info("📋 Refined plan:")
                for i, step in enumerate(refined_plan, 1):
                    logger.info(f"  {i}. {step.agent_name}: {step.title}")
                
                add_audit_event(state, "plan_refined", {
                    "iteration": state["refinement_count"],
                    "issues": quality_report.issues,
                    "new_steps": len(refined_plan)
                })
            else:
                # Accept result
                if quality_report.is_acceptable:
                    logger.info("✅ Quality acceptable - completing")
                    
                    # Save successful plan and episode to memory
                    self._save_to_memory(state, quality_report.score)
                else:
                    logger.warning(f"⚠️ Max refinements reached ({self.max_refinement_iterations}) - accepting result")
                
                state["status"] = "completed"
                state["next_agent"] = None
                
                add_audit_event(state, "quality_check_complete", {
                    "score": quality_report.score,
                    "acceptable": quality_report.is_acceptable,
                    "refinements": refinement_count
                })
        
        except Exception as e:
            logger.error(f"Error in quality evaluation: {e}", exc_info=True)
            # Accept result on error
            state["status"] = "completed"
            state["next_agent"] = None
        
        return state
    
    def _save_to_memory(self, state: AgentState, quality_score: float) -> None:
        """
        Save successful execution to memory.
        
        Args:
            state: Completed state
            quality_score: Quality score
        """
        try:
            plan = state.get("plan", {})
            plan_steps = plan.get("plan_steps", [])
            
            # Save plan to plan memory
            if plan_steps:
                from marie_agent.core.plan_generator import PlanStep
                steps = [PlanStep(**s) for s in plan_steps]
                
                self.plan_generator.save_successful_plan(
                    query=state["user_query"],
                    plan_steps=steps,
                    quality_score=quality_score
                )
            
            # Save episode to episodic memory
            if self.episodic_memory and state.get("final_answer"):
                self.episodic_memory.save_episode(
                    query=state["user_query"],
                    response=state["final_answer"],
                    plan_used=plan_steps,
                    success=True,
                    quality_score=quality_score,
                    user_feedback=None
                )
                
                logger.info("💾 Saved episode to memory")
            
        except Exception as e:
            logger.error(f"Error saving to memory: {e}")
    
    def route(self, state: AgentState) -> str:
        """
        Determine next agent to execute.
        
        Args:
            state: Current agent state
            
        Returns:
            Name of next agent node
        """
        # Check if waiting for human input
        if state["status"] == "waiting_human":
            return "human_interaction"
        
        # Check if completed or failed
        if state["status"] in ["completed", "failed"]:
            return "end"
        
        # Check if we have a next agent specified
        if state["next_agent"]:
            next_agent = state["next_agent"]
            logger.info(f"Routing to: {next_agent}")
            return next_agent
        
        # Determine next agent based on current step and plan
        if not state["plan"]:
            return "orchestrator"  # Need to plan first
        
        current_step = state["current_step"]
        
        if current_step >= len(state["plan"]["steps"]):
            # All steps completed
            state["status"] = "completed"
            return "end"
        
        # Map steps to agents
        agents = state["plan"]["agents_required"]
        if current_step < len(agents):
            return agents[current_step]
        
        return "end"
    
    def should_request_human_input(self, state: AgentState) -> bool:
        """
        Determine if human input is needed.
        
        Args:
            state: Current agent state
            
        Returns:
            True if human input required
        """
        # Check if any task has low confidence
        for task in state["tasks"]:
            if task["status"] == "needs_human":
                return True
            if task["confidence"] and task["confidence"] < self.confidence_threshold:
                return True
        
        # Check if plan requires human input
        if state["plan"] and state["plan"]["requires_human_input"]:
            return True
        
        # Check if explicitly flagged
        if state["requires_human_review"]:
            return True
        
        return False
    
    def evaluate_completion(self, state: AgentState) -> AgentState:
        """
        Evaluate if the goal has been achieved.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with completion evaluation
        """
        # Check all tasks completed successfully
        all_completed = all(
            task["status"] == "completed" 
            for task in state["tasks"]
        )
        
        # Check required outputs present
        has_answer = state["final_answer"] is not None
        has_evidence = len(state["evidence_map"]) > 0
        has_citations = len(state["citations"]) > 0
        
        if all_completed and has_answer and has_evidence:
            # Use LLM to assess confidence
            try:
                llm = get_llm_adapter()
                confidence_assessment = llm.assess_confidence(state)
                
                state["confidence_score"] = confidence_assessment["confidence_score"]
                state["confidence_assessment"] = confidence_assessment
                
                logger.info(
                    f"Confidence: {confidence_assessment['confidence_level']} "
                    f"({confidence_assessment['confidence_score']:.2%})"
                )
            except Exception as e:
                logger.warning(f"Could not assess confidence: {e}")
            
            state["status"] = "completed"
            state["next_agent"] = None
            
            add_audit_event(state, "request_completed", {
                "tasks_completed": len(state["tasks"]),
                "evidence_items": sum(len(ev) for ev in state["evidence_map"].values()),
                "citations": len(state["citations"]),
                "confidence": state.get("confidence_score", 0)
            })
        else:
            # Identify what's missing
            missing = []
            if not all_completed:
                missing.append("pending_tasks")
            if not has_answer:
                missing.append("final_answer")
            if not has_evidence:
                missing.append("evidence")
            
            logger.warning(f"Completion check failed. Missing: {missing}")
            
            add_audit_event(state, "completion_check_failed", {
                "missing": missing
            })
        
        return state


def orchestrator_node(state: AgentState) -> AgentState:
    """
    Orchestrator node function for LangGraph with Magentic architecture.
    
    Workflow:
    1. Planning Mode: Generate dynamic plan
    2. Execution Mode: Execute with progress tracking
    3. Quality Check: Evaluate and refine if needed
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state
    """
    orchestrator = OrchestratorAgent()
    
    # Check current state
    status = state.get("status", "planning")
    
    if status == "planning" or not state.get("plan"):
        # Planning Mode
        logger.info("=== PLANNING MODE ===")
        state = orchestrator.plan(state)
    
    elif status == "executing":
        # Execution Mode with progress tracking
        logger.info("=== EXECUTION MODE ===")
        state = orchestrator.execute_with_progress_tracking(state)
    
    elif status == "replanning":
        # Re-planning triggered by progress ledger
        logger.info("=== RE-PLANNING MODE ===")
        state["status"] = "planning"
        state = orchestrator.plan(state)
    
    elif status == "ready_for_quality_check":
        # Quality evaluation and potential refinement
        logger.info("=== QUALITY CHECK MODE ===")
        state = orchestrator.evaluate_quality_and_refine(state)
    
    else:
        # Completion check
        logger.info("=== EVALUATING COMPLETION ===")
        state = orchestrator.evaluate_completion(state)
    
    return state
