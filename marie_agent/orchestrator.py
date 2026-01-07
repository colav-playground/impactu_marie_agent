"""
Orchestrator Agent - Central control flow manager.

Based on Magentic Orchestration pattern.
"""

from typing import Dict, Any
import logging
from datetime import datetime

from marie_agent.state import AgentState, add_audit_event, Task
from marie_agent.adapters.llm_factory import get_llm_adapter

logger = logging.getLogger(__name__)


class OrchestratorAgent:
    """
    Central orchestrator managing the multi-agent workflow.
    
    Responsibilities:
    - Interpret user intent
    - Generate and adapt plans
    - Delegate tasks to specialized agents
    - Enforce evidence and confidence thresholds
    - Trigger human interaction when needed
    - Evaluate goal completion
    """
    
    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize orchestrator.
        
        Args:
            confidence_threshold: Minimum confidence for autonomous decisions
        """
        self.confidence_threshold = confidence_threshold
        
    def plan(self, state: AgentState) -> AgentState:
        """
        Create resolution plan for the user query.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with plan
        """
        logger.info(f"Planning for query: {state['user_query']}")
        
        try:
            # Use LLM to parse and plan query
            llm = get_llm_adapter()
            
            # Parse query to understand intent
            parsed_query = llm.parse_query(state["user_query"])
            logger.info(f"Query intent: {parsed_query.get('intent')}")
            
            # Create execution plan
            plan = llm.create_plan(state["user_query"], parsed_query)
            
            # Store in state
            state["plan"] = plan
            state["parsed_query"] = parsed_query
            state["status"] = "executing"
            state["next_agent"] = plan["agents_required"][0] if plan["agents_required"] else None
            
            add_audit_event(state, "plan_created", {
                "steps": len(plan["steps"]),
                "agents": plan["agents_required"],
                "intent": parsed_query.get("intent"),
                "complexity": plan.get("estimated_complexity")
            })
            
        except Exception as e:
            logger.error(f"Error during planning: {e}", exc_info=True)
            # Fallback to default plan
            state["plan"] = self._create_default_plan()
            state["next_agent"] = "entity_resolution"
        
        return state
    
    def _create_default_plan(self) -> Dict[str, Any]:
        """Create default fallback plan."""
        return {
            "steps": [
                "Resolve entities (authors, institutions)",
                "Retrieve evidence from MongoDB and OpenSearch",
                "Validate data consistency",
                "Compute metrics and analytics",
                "Build citations",
                "Generate report"
            ],
            "agents_required": [
                "entity_resolution",
                "retrieval",
                "validation",
                "metrics",
                "citations",
                "reporting"
            ],
            "estimated_complexity": "medium",
            "requires_human_input": False,
            "filters": {}
        }
    
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
    Orchestrator node function for LangGraph.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state
    """
    orchestrator = OrchestratorAgent()
    
    # If no plan, create one
    if not state["plan"]:
        state = orchestrator.plan(state)
    else:
        # Check for completion
        state = orchestrator.evaluate_completion(state)
    
    return state
