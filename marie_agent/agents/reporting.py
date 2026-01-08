"""
Reporting Agent - Generate final report and answer.

Responsible for:
- Producing executive summary
- Creating structured reports
- Documenting methods and filters
- Stating limitations
- Generating final answer
"""

from typing import Dict, Any, List
import logging
from datetime import datetime

from marie_agent.state import AgentState, add_audit_event

logger = logging.getLogger(__name__)


class ReportingAgent:
    """
    Agent responsible for generating final reports.
    
    Produces structured, evidence-backed answers.
    """
    
    def generate_report(self, state: AgentState) -> AgentState:
        """
        Generate final report and answer.
        Handles both research-based (RAG) and direct LLM responses.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with final answer and report
        """
        logger.info("Generating final report")
        
        try:
            query = state["user_query"]
            needs_rag = state.get("needs_rag", True)
            
            if not needs_rag:
                # Direct LLM response for general knowledge queries
                logger.info("Generating direct response (no RAG)")
                answer = self._generate_direct_answer(query, state)
                
                state["final_answer"] = answer
                state["response_mode"] = "direct"
                state["status"] = "completed"
                
                add_audit_event(state, "report_generated", {
                    "mode": "direct_llm",
                    "needs_rag": False,
                    "answer_length": len(answer)
                })
                
            else:
                # Research-based response with RAG
                logger.info("Generating research report (with RAG)")
                entities = state.get("entities_resolved", {})
                metrics = state.get("computed_metrics", {})
                citations = state.get("citations", [])
                
                # Build answer
                answer = self._build_answer(query, entities, metrics)
                
                # Build full report
                report = self._build_report(query, entities, metrics, citations)
            
            # Calculate overall confidence
            confidence = self._calculate_confidence(state)
            
            state["final_answer"] = answer
            state["report"] = report
            state["confidence_score"] = confidence
            
            add_audit_event(state, "report_generated", {
                "answer_length": len(answer),
                "report_length": len(report),
                "confidence": confidence
            })
            
            state["next_agent"] = None
            state["status"] = "completed"
            
            logger.info("Report generated successfully")
            
        except Exception as e:
            logger.error(f"Error generating report: {e}", exc_info=True)
            state["error"] = str(e)
            state["next_agent"] = None
            state["status"] = "failed"
        
        return state
    
    def _build_answer(
        self,
        query: str,
        entities: Dict[str, Any],
        metrics: Dict[str, Any]
    ) -> str:
        """Build concise answer to user query."""
        
        top_cited = metrics.get("top_cited", [])
        institutions = entities.get("institutions", [])
        
        # Build institution context
        inst_context = ""
        if institutions:
            inst_names = ", ".join([i["name"] for i in institutions[:2]])
            inst_context = f" from {inst_names}"
        
        # Build top papers list
        if not top_cited:
            return f"No papers found{inst_context} matching your query."
        
        answer_parts = [
            f"## Top {len(top_cited)} Most Cited Papers{inst_context}\n"
        ]
        
        for i, paper in enumerate(top_cited, 1):
            title = paper.get("title", "Unknown title")
            year = paper.get("year", "Unknown")
            citations = paper.get("citations", 0)
            doi = paper.get("doi", "")
            
            # Format citations (handle None)
            citations_str = f"{citations:,}" if citations is not None else "0"
            
            answer_parts.append(
                f"\n**{i}. {title}**\n"
                f"   - Year: {year}\n"
                f"   - Citations: {citations_str}\n"
            )
            if doi:
                answer_parts.append(f"   - DOI: {doi}\n")
        
        # Add summary stats
        total_docs = metrics.get("total_documents", 0)
        citation_stats = metrics.get("citation_stats", {})
        
        if citation_stats.get("available"):
            answer_parts.append(
                f"\n### Summary Statistics\n"
                f"- Total documents analyzed: {total_docs}\n"
                f"- Total citations: {citation_stats.get('total', 0):,}\n"
                f"- Average citations: {citation_stats.get('mean', 0):.1f}\n"
            )
        
        return "".join(answer_parts)
    
    def _build_report(
        self,
        query: str,
        entities: Dict[str, Any],
        metrics: Dict[str, Any],
        citations: List[Dict[str, Any]]
    ) -> str:
        """Build full structured report."""
        
        report_parts = [
            "# MARIE Analysis Report\n\n",
            f"**Query:** {query}\n\n",
            f"**Generated:** {datetime.utcnow().isoformat()}\n\n",
            "---\n\n"
        ]
        
        # Executive summary
        report_parts.append("## Executive Summary\n\n")
        report_parts.append(self._build_answer(query, entities, metrics))
        report_parts.append("\n\n---\n\n")
        
        # Methods
        report_parts.append("## Methods\n\n")
        report_parts.append("**Data Sources:**\n")
        report_parts.append("- MongoDB (structured bibliographic data)\n")
        report_parts.append("- OpenSearch (full-text search with RAG)\n\n")
        
        report_parts.append("**Entity Resolution:**\n")
        institutions = entities.get("institutions", [])
        if institutions:
            for inst in institutions:
                report_parts.append(
                    f"- {inst['name']} (confidence: {inst['confidence']:.2f})\n"
                )
        report_parts.append("\n")
        
        # Metrics
        report_parts.append("## Detailed Metrics\n\n")
        by_year = metrics.get("by_year", {})
        if by_year:
            report_parts.append("**Publications by Year:**\n")
            for year, count in sorted(by_year.items(), reverse=True)[:5]:
                report_parts.append(f"- {year}: {count}\n")
            report_parts.append("\n")
        
        by_type = metrics.get("by_type", {})
        if by_type:
            report_parts.append("**Publications by Type:**\n")
            for ptype, count in by_type.items():
                report_parts.append(f"- {ptype}: {count}\n")
            report_parts.append("\n")
        
        # Citations
        report_parts.append("## References\n\n")
        report_parts.append(f"Total citations: {len(citations)}\n\n")
        report_parts.append("Evidence sources:\n")
        for i, citation in enumerate(citations[:10], 1):  # Limit to first 10
            report_parts.append(
                f"{i}. {citation.get('reference')} "
                f"(confidence: {citation.get('confidence', 0):.2f})\n"
            )
        
        # Limitations
        report_parts.append("\n\n## Limitations\n\n")
        report_parts.append(
            "- Results limited to available indexed data\n"
            "- Citation counts may not reflect real-time data\n"
            "- Entity resolution confidence varies by data quality\n"
        )
        
        return "".join(report_parts)
    
    def _generate_direct_answer(self, query: str, state: AgentState) -> str:
        """
        Generate direct LLM answer for general knowledge queries.
        No RAG, no citations, clean explanation.
        
        Args:
            query: User question
            state: Current state (for context if needed)
            
        Returns:
            Direct answer string
        """
        from marie_agent.adapters.llm_factory import get_llm_adapter
        
        try:
            llm = get_llm_adapter()
            
            # Get conversation context if available
            conversation_history = state.get("conversation_context", [])
            
            # Generate clean, direct answer
            answer = f"""## {query}

{llm.generate_text(f"Answer this question clearly and concisely: {query}")}

---
*This is a general knowledge response. For research-specific queries about institutions, authors, or papers, please ask about specific entities.*
"""
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating direct answer: {e}")
            return f"## Response\n\nI understand your question about: {query}\n\nHowever, I encountered an issue generating a response. Could you rephrase your question or ask something more specific?"
    
    def _calculate_confidence(self, state: AgentState) -> float:
        """Calculate overall confidence score."""
        
        # Average confidence from citations
        citations = state.get("citations", [])
        if not citations:
            return 0.0
        
        confidences = [c.get("confidence", 0.0) for c in citations]
        avg_confidence = sum(confidences) / len(confidences)
        
        # Adjust based on data availability
        metrics = state.get("computed_metrics", {})
        if metrics.get("total_documents", 0) < 5:
            avg_confidence *= 0.8  # Reduce confidence if few documents
        
        return min(avg_confidence, 1.0)


def reporting_agent_node(state: AgentState) -> AgentState:
    """
    Reporting agent node function for LangGraph.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with final report
    """
    agent = ReportingAgent()
    return agent.generate_report(state)
