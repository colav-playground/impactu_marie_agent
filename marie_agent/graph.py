"""
LangGraph workflow for MARIE multi-agent system.
"""

from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END
import logging

from marie_agent.state import AgentState, create_initial_state
from marie_agent.orchestrator import orchestrator_node, OrchestratorAgent
from marie_agent.agents.entity_resolution import entity_resolution_agent_node
from marie_agent.agents.retrieval import retrieval_agent_node
from marie_agent.agents.metrics import metrics_agent_node
from marie_agent.agents.citations import citations_agent_node
from marie_agent.agents.reporting import reporting_agent_node
from marie_agent.memory import get_memory_manager

logger = logging.getLogger(__name__)


def create_marie_graph() -> StateGraph:
    """
    Create the MARIE multi-agent workflow graph.
    
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
    workflow.add_node("metrics", metrics_agent_node)
    workflow.add_node("citations", citations_agent_node)
    workflow.add_node("reporting", reporting_agent_node)
    
    # Placeholder for validation (simple passthrough for now)
    workflow.add_node("validation", lambda state: {**state, "next_agent": "metrics"})
    
    # Set entry point
    workflow.set_entry_point("orchestrator")
    
    # Add routing logic
    orchestrator = OrchestratorAgent()
    
    workflow.add_conditional_edges(
        "orchestrator",
        orchestrator.route,
        {
            "entity_resolution": "entity_resolution",
            "retrieval": "retrieval",
            "validation": "validation",
            "metrics": "metrics",
            "citations": "citations",
            "reporting": "reporting",
            "end": END
        }
    )
    
    # Each agent routes back to orchestrator for next decision
    for agent_node in ["entity_resolution", "retrieval", "validation", "metrics", "citations", "reporting"]:
        workflow.add_edge(agent_node, "orchestrator")
    
    # Compile graph
    app = workflow.compile()
    
    return app


def run_marie_query(
    user_query: str,
    request_id: str,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Execute a query through the MARIE multi-agent system.
    
    Args:
        user_query: User's question
        request_id: Unique request identifier
        session_id: Optional session ID for multi-turn conversations
        user_id: Optional user ID for personalization
        
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
    
    # Create and run graph
    app = create_marie_graph()
    
    # Execute workflow
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
