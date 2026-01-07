"""
Citations Agent - Build citation mapping for claims.

Responsible for:
- Mapping every claim to explicit evidence
- Creating traceable references
- Enforcing citation completeness
- Generating reference list
"""

from typing import Dict, Any, List
import logging
from datetime import datetime

from marie_agent.state import AgentState, add_audit_event

logger = logging.getLogger(__name__)


class CitationsAgent:
    """
    Agent responsible for building citations.
    
    Ensures every claim is backed by evidence.
    """
    
    def build_citations(self, state: AgentState) -> AgentState:
        """
        Build citation map for all claims.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with citations
        """
        logger.info("Building citations for evidence")
        
        try:
            evidence_map = state.get("evidence_map", {})
            metrics = state.get("computed_metrics", {})
            
            citations = []
            
            # Citation for retrieval results
            if "opensearch_results" in evidence_map:
                for evidence in evidence_map["opensearch_results"]:
                    citation = {
                        "claim_id": "retrieved_documents",
                        "source_type": evidence["source_type"],
                        "reference": evidence["reference"],
                        "confidence": evidence["confidence"],
                        "timestamp": evidence["timestamp"]
                    }
                    citations.append(citation)
            
            # Citation for top cited papers
            top_cited = metrics.get("top_cited", [])
            for i, paper in enumerate(top_cited):
                citation = {
                    "claim_id": f"top_cited_{i+1}",
                    "source_type": "mongodb",
                    "reference": f"mongodb:works/{paper.get('id')}",
                    "title": paper.get("title"),
                    "doi": paper.get("doi"),
                    "citations": paper.get("citations"),
                    "confidence": 1.0  # Direct from database
                }
                citations.append(citation)
            
            state["citations"] = citations
            
            add_audit_event(state, "citations_built", {
                "total_citations": len(citations),
                "evidence_sources": len(evidence_map)
            })
            
            state["next_agent"] = "reporting"
            
            logger.info(f"Built {len(citations)} citations")
            
        except Exception as e:
            logger.error(f"Error building citations: {e}", exc_info=True)
            state["error"] = str(e)
            state["next_agent"] = None
            state["status"] = "failed"
        
        return state


def citations_agent_node(state: AgentState) -> AgentState:
    """
    Citations agent node function for LangGraph.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with citations
    """
    agent = CitationsAgent()
    return agent.build_citations(state)
