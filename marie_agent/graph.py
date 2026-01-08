"""
LangGraph workflow for MARIE multi-agent system.

Enhanced with:
- Intelligent conditional routing
- Error handling and retry logic
- Checkpointing support
- Better state management
"""

from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import logging

from marie_agent.state import AgentState, create_initial_state
from marie_agent.orchestrator import orchestrator_node, OrchestratorAgent
from marie_agent.agents.entity_resolution import entity_resolution_agent_node
from marie_agent.agents.retrieval import retrieval_agent_node
from marie_agent.agents.metrics import metrics_agent_node
from marie_agent.agents.citations import citations_agent_node
from marie_agent.agents.reporting import reporting_agent_node
from marie_agent.agents.opensearch_expert import opensearch_expert_node
from marie_agent.memory import get_memory_manager
from marie_agent.human_interaction import get_human_manager
from marie_agent.core.routing import should_continue, route_after_error
from marie_agent.core.exceptions import AgentExecutionError

logger = logging.getLogger(__name__)


def human_interaction_node(state: AgentState) -> AgentState:
    """
    Handle human-in-the-loop interactions.
    
    Processes pending human interactions and waits for input when needed.
    """
    logger.info("Human interaction node activated")
    
    human_manager = get_human_manager()
    
    # Get pending interactions
    pending = [hi for hi in state.get("human_interactions", []) if hi["status"] == "pending"]
    
    if not pending:
        logger.warning("No pending human interactions found")
        return {**state, "next_agent": "orchestrator"}
    
    # Process the first pending interaction
    interaction = pending[0]
    logger.info(f"Processing {interaction['type']} interaction: {interaction['interaction_id']}")
    
    # For now, auto-approve in non-interactive mode
    # In production, this would wait for actual human input
    if interaction["type"] == "co_planning":
        # Auto-approve the plan
        response = "Approved - proceed with the plan"
        logger.info("Auto-approving co-planning request (non-interactive mode)")
        
    elif interaction["type"] == "action_guard":
        # Auto-approve the action
        response = "Approved"
        logger.info("Auto-approving action guard (non-interactive mode)")
        
    elif interaction["type"] == "verification":
        # Auto-confirm verification
        response = "Verified"
        logger.info("Auto-confirming verification (non-interactive mode)")
        
    else:
        response = "Approved"
        logger.info(f"Auto-approving {interaction['type']} (non-interactive mode)")
    
    # Update interaction status
    updated_interactions = []
    for hi in state.get("human_interactions", []):
        if hi["interaction_id"] == interaction["interaction_id"]:
            hi["status"] = "completed"
            hi["response"] = response
        updated_interactions.append(hi)
    
    # Update state
    new_state = {
        **state,
        "human_interactions": updated_interactions,
        "requires_human_review": False,
        "next_agent": "orchestrator"
    }
    
    logger.info("Human interaction completed, returning to orchestrator")
    return new_state


def error_handler_node(state: AgentState) -> AgentState:
    """
    Handle errors and decide on recovery strategy.
    
    Implements:
    - Error logging
    - Retry logic
    - Graceful degradation
    - Partial result recovery
    """
    logger.error("Error handler node activated")
    
    error = state.get("error", "Unknown error")
    failed_agent = state.get("failed_agent", "unknown")
    retry_count = state.get("retry_count", 0)
    
    logger.error(f"Error in {failed_agent}: {error} (retry {retry_count})")
    
    # Increment retry count
    new_retry_count = retry_count + 1
    
    # Determine if error is recoverable
    recoverable = True
    if "connection" in str(error).lower() or "timeout" in str(error).lower():
        recoverable = True
        logger.info("Error appears recoverable (connection/timeout)")
    elif "validation" in str(error).lower():
        recoverable = False
        logger.warning("Validation error - not recoverable")
    
    # Update state
    new_state = {
        **state,
        "retry_count": new_retry_count,
        "recoverable": recoverable,
        "error_handled": True
    }
    
    # Clear error for retry
    if new_retry_count < 3:
        new_state["error"] = None
        new_state["failed_agent"] = None
        logger.info(f"Cleared error for retry #{new_retry_count}")
    else:
        logger.error("Max retries reached - marking as unrecoverable")
        new_state["recoverable"] = False
    
    return new_state


def create_marie_graph(enable_checkpointing: bool = False) -> StateGraph:
    """
    Create the MARIE multi-agent workflow graph with enhanced features.
    
    Features:
    - Intelligent conditional routing
    - Error handling and retry logic
    - Optional checkpointing for fault tolerance
    - Better state management
    
    Args:
        enable_checkpointing: Enable checkpointing for persistence
    
    Returns:
        Compiled StateGraph
    """
    # Initialize graph with state schema
    workflow = StateGraph(AgentState)
    
    # Add orchestrator node
    workflow.add_node("orchestrator", orchestrator_node)
    
    # Add specialized agent nodes
    workflow.add_node("entity_resolution", entity_resolution_agent_node)
    workflow.add_node("retrieval", retrieval_agent_node)
    workflow.add_node("opensearch_expert", opensearch_expert_node)
    workflow.add_node("metrics", metrics_agent_node)
    workflow.add_node("citations", citations_agent_node)
    workflow.add_node("reporting", reporting_agent_node)
    
    # Add support nodes
    workflow.add_node("human_interaction", human_interaction_node)
    workflow.add_node("error_handler", error_handler_node)
    workflow.add_node("validation", lambda state: {**state, "next_agent": "metrics"})
    
    # Set entry point
    workflow.set_entry_point("orchestrator")
    
    # Add intelligent conditional routing from orchestrator
    orchestrator = OrchestratorAgent()
    
    workflow.add_conditional_edges(
        "orchestrator",
        orchestrator.route,
        {
            "entity_resolution": "entity_resolution",
            "retrieval": "retrieval",
            "opensearch_expert": "opensearch_expert",
            "validation": "validation",
            "metrics": "metrics",
            "citations": "citations",
            "reporting": "reporting",
            "human_interaction": "human_interaction",
            "error_handler": "error_handler",
            "end": END
        }
    )
    
    # Add conditional routing from error handler
    workflow.add_conditional_edges(
        "error_handler",
        route_after_error,
        {
            "entity_resolution": "entity_resolution",
            "retrieval": "retrieval",
            "opensearch_expert": "opensearch_expert",
            "metrics": "metrics",
            "reporting": "reporting",
            "end": END
        }
    )
    
    # Each agent routes back to orchestrator for next decision
    agent_nodes = [
        "entity_resolution", 
        "retrieval", 
        "opensearch_expert", 
        "validation", 
        "metrics", 
        "citations", 
        "reporting", 
        "human_interaction"
    ]
    
    for agent_node in agent_nodes:
        # Use conditional routing to check for errors
        workflow.add_conditional_edges(
            agent_node,
            should_continue,
            {
                "entity_resolution": "entity_resolution",
                "retrieval": "retrieval",
                "opensearch_expert": "opensearch_expert",
                "validation": "validation",
                "metrics": "metrics",
                "citations": "citations",
                "reporting": "reporting",
                "human_interaction": "human_interaction",
                "error_handler": "error_handler",
                "end": END
            }
        )
    
    # Compile graph with optional checkpointing
    if enable_checkpointing:
        # Use memory saver for checkpointing
        checkpointer = MemorySaver()
        app = workflow.compile(checkpointer=checkpointer)
        logger.info("Graph compiled with checkpointing enabled")
    else:
        app = workflow.compile()
        logger.info("Graph compiled without checkpointing")
    
    return app


def run_marie_query(
    user_query: str,
    request_id: str,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    enable_checkpointing: bool = False
) -> Dict[str, Any]:
    """
    Execute a query through the MARIE multi-agent system.
    
    Args:
        user_query: User's question
        request_id: Unique request identifier
        session_id: Optional session ID for multi-turn conversations
        user_id: Optional user ID for personalization
        enable_checkpointing: Enable fault-tolerant checkpointing
        
    Returns:
        Final state with answer and evidence
    """
    logger.info(f"Starting MARIE query: {user_query}")
    
    # Initialize memory manager
    memory = get_memory_manager()
    
    # Get or create session
    if session_id:
        session = memory.get_session(session_id)
        if not session:
            session = memory.create_session(session_id, user_id)
    else:
        session = memory.create_session(request_id, user_id)
        session_id = request_id
    
    # Get conversation context
    conversation_context = memory.get_conversation_context(session_id, turns=3)
    
    # Create initial state
    initial_state = create_initial_state(user_query, request_id)
    
    # Add conversation context to state
    if conversation_context:
        initial_state["parsed_query"] = {
            "conversation_history": conversation_context
        }
        logger.info(f"Loaded {len(conversation_context)} previous turns")
    
    # Create and run graph with checkpointing option
    app = create_marie_graph(enable_checkpointing=enable_checkpointing)
    
    # Execute workflow (with config if checkpointing enabled)
    if enable_checkpointing:
        config = {"configurable": {"thread_id": session_id}}
        final_state = app.invoke(initial_state, config=config)
    else:
        final_state = app.invoke(initial_state)
    
    # Record turn in memory
    memory.record_turn(
        session_id=session_id,
        turn_id=request_id,
        user_query=user_query,
        agent_response=final_state.get("final_answer", ""),
        confidence=final_state.get("confidence_score", 0.0),
        evidence_count=len(final_state.get("evidence_map", {})),
        metadata={
            "status": final_state["status"],
            "agents_used": len([t for t in final_state.get("tasks", []) if t["status"] == "completed"])
        }
    )
    
    # Persist session
    memory.save_session(session_id)
    
    logger.info(f"MARIE query completed. Status: {final_state['status']}")
    
    # Add session info to response
    final_state["session_id"] = session_id
    final_state["turn_count"] = len(session.turns) if session else 1
    
    return final_state

def stream_marie_query(
    user_query: str,
    request_id: str,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None
):
    """
    Stream execution of a MARIE query (LangGraph best practice).
    
    Yields state updates as they happen for responsive UX.
    
    Args:
        user_query: User's question
        request_id: Unique request identifier
        session_id: Optional session ID
        user_id: Optional user ID
        
    Yields:
        State updates from each agent execution
    """
    logger.info(f"Starting MARIE query (streaming): {user_query}")
    
    # Initialize memory manager
    memory = get_memory_manager()
    
    # Get or create session
    if session_id:
        session = memory.get_session(session_id)
        if not session:
            session = memory.create_session(session_id, user_id)
    else:
        session = memory.create_session(request_id, user_id)
        session_id = request_id
    
    # Get conversation context
    conversation_context = memory.get_conversation_context(session_id, turns=3)
    
    # Create initial state
    initial_state = create_initial_state(user_query, request_id)
    
    # Add conversation context
    if conversation_context:
        initial_state["parsed_query"] = {
            "conversation_history": conversation_context
        }
    
    # Create and run graph
    app = create_marie_graph()
    
    # Stream execution
    final_state = None
    for step_output in app.stream(initial_state, stream_mode="values"):
        final_state = step_output
        yield step_output
    
    # Record turn in memory
    if final_state:
        memory.record_turn(
            session_id=session_id,
            turn_id=request_id,
            user_query=user_query,
            agent_response=final_state.get("final_answer", ""),
            confidence=final_state.get("confidence_score", 0.0),
            evidence_count=len(final_state.get("evidence_map", {})),
            metadata={
                "status": final_state["status"],
                "agents_used": len([t for t in final_state.get("tasks", []) if t["status"] == "completed"])
            }
        )
        
        # Persist session
        memory.save_session(session_id)
        
        logger.info(f"MARIE query completed (streaming). Status: {final_state['status']}")