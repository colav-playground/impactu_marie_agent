"""
Enhanced routing logic for LangGraph workflow.

Provides intelligent conditional routing based on state.
"""

from typing import Literal
import logging

from marie_agent.state import AgentState
from marie_agent.config import AgentConstants

logger = logging.getLogger(__name__)

# Type for agent routing
AgentRoute = Literal[
    "entity_resolution",
    "retrieval", 
    "opensearch_expert",
    "validation",
    "metrics",
    "citations",
    "reporting",
    "human_interaction",
    "error_handler",
    "end"
]


def should_continue(state: AgentState) -> AgentRoute:
    """
    Intelligent routing decision based on current state.
    
    Implements smart conditional edges:
    - Error handling routing
    - Human interaction when needed
    - Skip unnecessary agents
    - End when complete
    
    Args:
        state: Current agent state
        
    Returns:
        Next agent to route to
    """
    # Check for errors first
    if state.get("error") or state.get("failed_agent"):
        logger.warning(f"Error detected: {state.get('error')}")
        return "error_handler"
    
    # Check if human interaction is required
    if state.get("requires_human_review"):
        pending = [hi for hi in state.get("human_interactions", []) 
                   if hi["status"] == "pending"]
        if pending:
            logger.info("Human interaction required")
            return "human_interaction"
    
    # Check completion
    if state.get("final_answer") and state.get("status") == "completed":
        logger.info("Query completed successfully")
        return "end"
    
    # Route based on plan/current_step
    plan = state.get("plan")
    if not plan or not plan.get("steps"):
        logger.info("No plan - going to orchestrator")
        return "end"
    
    current_step = state.get("current_step", 0)
    steps = plan.get("steps", [])
    
    if current_step >= len(steps):
        logger.info("All steps completed - finishing")
        return "reporting"
    
    # Get current step action
    step = steps[current_step]
    step_lower = step.lower()
    
    # Route based on step content
    if "resolve" in step_lower or "entity" in step_lower or "institution" in step_lower:
        return "entity_resolution"
    elif "retrieve" in step_lower or "search" in step_lower or "query" in step_lower:
        # Check if we need OpenSearch expert
        if "opensearch" in step_lower or "index" in step_lower:
            return "opensearch_expert"
        return "retrieval"
    elif "metric" in step_lower or "calculate" in step_lower or "compute" in step_lower:
        return "metrics"
    elif "cite" in step_lower or "reference" in step_lower:
        return "citations"
    elif "report" in step_lower or "answer" in step_lower or "respond" in step_lower:
        return "reporting"
    
    # Default: go back to orchestrator for replanning
    logger.warning(f"Unclear routing for step: {step}")
    return "end"


def should_retry(state: AgentState) -> bool:
    """
    Determine if agent should retry after failure.
    
    Args:
        state: Current state
        
    Returns:
        True if should retry
    """
    # Get retry count
    retry_count = state.get("retry_count", 0)
    max_retries = 2
    
    # Check if there's an error and we haven't exceeded retries
    has_error = bool(state.get("error") or state.get("failed_agent"))
    
    return has_error and retry_count < max_retries


def route_after_error(state: AgentState) -> AgentRoute:
    """
    Route after error handling.
    
    Args:
        state: Current state
        
    Returns:
        Next agent route
    """
    if should_retry(state):
        # Retry the failed agent
        failed_agent = state.get("failed_agent")
        if failed_agent:
            logger.info(f"Retrying failed agent: {failed_agent}")
            return failed_agent
    
    # Check if error is recoverable
    if state.get("recoverable", False):
        # Try to continue with partial results
        return "reporting"
    
    # Unrecoverable error
    logger.error("Unrecoverable error - ending workflow")
    return "end"


def should_use_opensearch_expert(state: AgentState) -> bool:
    """
    Determine if OpenSearch expert is needed.
    
    Args:
        state: Current state
        
    Returns:
        True if OpenSearch expert should be used
    """
    query = state.get("user_query", "").lower()
    
    # Keywords that indicate need for OpenSearch expert
    expert_keywords = [
        "opensearch",
        "index",
        "mapping",
        "complex query",
        "aggregation",
        "analytics"
    ]
    
    return any(keyword in query for keyword in expert_keywords)


def compute_confidence_threshold(state: AgentState) -> float:
    """
    Compute dynamic confidence threshold based on query complexity.
    
    Args:
        state: Current state
        
    Returns:
        Confidence threshold
    """
    plan = state.get("plan")
    if not plan:
        return AgentConstants.DEFAULT_CONFIDENCE
    
    complexity = plan.get("estimated_complexity", "medium")
    
    # Lower threshold for complex queries (we're doing our best)
    # Higher threshold for simple queries (should be confident)
    thresholds = {
        "simple": 0.85,
        "medium": 0.75,
        "complex": 0.65
    }
    
    return thresholds.get(complexity, AgentConstants.DEFAULT_CONFIDENCE)


def should_seek_human_approval(state: AgentState) -> bool:
    """
    Determine if human approval is needed.
    
    Args:
        state: Current state
        
    Returns:
        True if human approval needed
    """
    # Check confidence
    confidence = state.get("confidence_score", 0.0)
    threshold = compute_confidence_threshold(state)
    
    if confidence < threshold:
        logger.info(f"Confidence {confidence:.2f} below threshold {threshold:.2f}")
        return True
    
    # Check if plan requires human input
    plan = state.get("plan")
    if plan and plan.get("requires_human_input"):
        return True
    
    # Check for high-risk actions
    tasks = state.get("tasks", [])
    for task in tasks:
        if task.get("risk_level") == "high":
            return True
    
    return False
