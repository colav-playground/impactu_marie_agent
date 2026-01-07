"""
Metrics Agent - Compute scientometric indicators.

Responsible for:
- Calculating productivity metrics
- Computing impact indicators
- Aggregating temporal data
- Generating statistical summaries
"""

from typing import Dict, Any, List
import logging
from datetime import datetime
from collections import defaultdict

from marie_agent.state import AgentState, add_audit_event

logger = logging.getLogger(__name__)


class MetricsAgent:
    """
    Agent responsible for computing metrics and analytics.
    
    Only computes data-supported indicators.
    """
    
    def compute(self, state: AgentState) -> AgentState:
        """
        Compute metrics from retrieved data.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with computed metrics
        """
        logger.info("Computing metrics from retrieved data")
        
        try:
            retrieved_data = state.get("retrieved_data", [])
            
            if not retrieved_data:
                logger.warning("No data to compute metrics from")
                state["next_agent"] = "citations"
                return state
            
            # Compute various metrics
            metrics = {
                "total_documents": len(retrieved_data),
                "by_type": self._count_by_type(retrieved_data),
                "by_year": self._count_by_year(retrieved_data),
                "citation_stats": self._compute_citation_stats(retrieved_data),
                "top_cited": self._get_top_cited(retrieved_data, limit=5)
            }
            
            state["computed_metrics"] = metrics
            
            add_audit_event(state, "metrics_computed", {
                "total_documents": metrics["total_documents"],
                "unique_years": len(metrics["by_year"]),
                "top_cited_count": len(metrics["top_cited"])
            })
            
            state["next_agent"] = "citations"
            
            logger.info(f"Computed metrics for {metrics['total_documents']} documents")
            
        except Exception as e:
            logger.error(f"Error computing metrics: {e}", exc_info=True)
            state["error"] = str(e)
            state["next_agent"] = None
            state["status"] = "failed"
        
        return state
    
    def _count_by_type(self, data: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count documents by type."""
        counts = defaultdict(int)
        for doc in data:
            work_type = doc.get("source", {}).get("work_type", "unknown")
            counts[work_type] += 1
        return dict(counts)
    
    def _count_by_year(self, data: List[Dict[str, Any]]) -> Dict[int, int]:
        """Count documents by year."""
        counts = defaultdict(int)
        for doc in data:
            year = doc.get("source", {}).get("year")
            if year:
                counts[int(year)] += 1
        return dict(sorted(counts.items()))
    
    def _compute_citation_stats(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute citation statistics."""
        citations = []
        for doc in data:
            cites = doc.get("source", {}).get("citations_count")
            if cites is not None:
                citations.append(cites)
        
        if not citations:
            return {"available": False}
        
        return {
            "available": True,
            "total": sum(citations),
            "mean": sum(citations) / len(citations),
            "max": max(citations),
            "min": min(citations)
        }
    
    def _get_top_cited(self, data: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
        """Get top cited documents."""
        # Sort by citations
        sorted_docs = sorted(
            data,
            key=lambda x: x.get("source", {}).get("citations_count", 0),
            reverse=True
        )
        
        top_docs = []
        for doc in sorted_docs[:limit]:
            source = doc.get("source", {})
            top_docs.append({
                "id": doc.get("id"),
                "title": source.get("title", "Unknown"),
                "year": source.get("year"),
                "citations": source.get("citations_count"),
                "authors": source.get("authors", "")[:100],  # Truncate
                "doi": source.get("doi")
            })
        
        return top_docs


def metrics_agent_node(state: AgentState) -> AgentState:
    """
    Metrics agent node function for LangGraph.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with computed metrics
    """
    agent = MetricsAgent()
    return agent.compute(state)
