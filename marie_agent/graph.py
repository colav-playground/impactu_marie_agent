"""
LangGraph workflow for MARIE multi-agent system.
"""

from typing import Dict, Any
from langgraph.graph import StateGraph, END
import logging

from marie_agent.state import AgentState, create_initial_state
from marie_agent.orchestrator import orchestrator_node, OrchestratorAgent
from marie_agent.agents.entity_resolution import entity_resolution_agent_node
from marie_agent.agents.retrieval import retrieval_agent_node
from marie_agent.agents.metrics import metrics_agent_node
from marie_agent.agents.citations import citations_agent_node
from marie_agent.agents.reporting import reporting_agent_node

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


def run_marie_query(user_query: str, request_id: str) -> Dict[str, Any]:
    """
    Execute a query through the MARIE multi-agent system.
    
    Args:
        user_query: User's question
        request_id: Unique request identifier
        
    Returns:
        Final state with answer and evidence
    """
    logger.info(f"Starting MARIE query: {user_query}")
    
    # Create initial state
    initial_state = create_initial_state(user_query, request_id)
    
    # Create and run graph
    app = create_marie_graph()
    
    # Execute workflow
    final_state = app.invoke(initial_state)
    
    logger.info(f"MARIE query completed. Status: {final_state['status']}")
    
    return final_state
