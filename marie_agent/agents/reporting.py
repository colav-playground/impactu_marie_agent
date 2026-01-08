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
        Generates natural, contextual responses based on query type.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with final answer and report
        """
        logger.info("Generating final report")
        
        try:
            query = state["user_query"]
            
            # Always generate natural responses
            logger.info("Generating natural contextual answer")
            answer = self._generate_natural_answer(query, state)
            report = answer
            response_mode = "natural"
            
            # Calculate overall confidence
            confidence = self._calculate_confidence(state)
            
            state["final_answer"] = answer
            state["report"] = report
            state["confidence_score"] = confidence
            state["response_mode"] = response_mode
            
            add_audit_event(state, "report_generated", {
                "mode": response_mode,
                "answer_length": len(answer),
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
    
    def _generate_natural_answer(self, query: str, state: AgentState) -> str:
        """
        Generate natural language answer using Prompt Engineer Service.
        """
        from marie_agent.adapters.llm_factory import create_llm_adapter
        from marie_agent.agents.prompt_engineer import get_prompt_engineer
        
        # Get retrieved documents and metrics
        evidence_map = state.get("evidence_map", {})
        documents = evidence_map.get("evidence", [])
        metrics = state.get("computed_metrics", {})
        entities = state.get("entities_resolved", {})
        
        # Prepare context for prompt engineer
        context = {
            "query": query,
            "documents": documents,
            "metrics": metrics,
            "entities": entities,
            "has_sources": len(documents) > 0
        }
        
        # Get prompt engineer service
        prompt_engineer = get_prompt_engineer()
        
        # Build context from documents and metrics FIRST
        context_parts = []
        
        # Add entity context
        institutions = entities.get("institutions", [])
        if institutions:
            inst_names = ", ".join([i["name"] for i in institutions[:2]])
            context_parts.append(f"Institución(es): {inst_names}")
        
        # Add document info
        context_parts.append(f"\nDocumentos encontrados: {len(documents)}")
        
        # Add top papers with details
        if documents:
            context_parts.append("\nPapers más relevantes:")
            for i, doc in enumerate(documents[:10], 1):
                title = doc.get("title", "Unknown")
                year = doc.get("year", "")
                citations = doc.get("citations", 0)
                authors = doc.get("authors", [])
                
                author_str = ""
                if authors:
                    author_names = [a.get("full_name", "") for a in authors[:3]]
                    author_str = f" - Autores: {', '.join(filter(None, author_names))}"
                
                context_parts.append(f"{i}. {title} ({year}) - {citations} citas{author_str}")
        
        # Add metrics
        if metrics:
            citation_stats = metrics.get("citation_stats", {})
            total_citations = citation_stats.get("total", 0)
            avg_citations = citation_stats.get("mean", 0)
            if total_citations > 0:
                context_parts.append(f"\nCitas totales: {total_citations}, Promedio: {avg_citations:.1f}")
        
        # Build enriched context
        data_context = "\n".join(context_parts)
        context["data_summary"] = data_context
        
        # Build optimized prompt using Chain-of-Thought for synthesis
        task_description = f"Synthesize information from {len(documents)} documents and generate a comprehensive answer based on REAL DATA"
        optimized_prompt = prompt_engineer.build_prompt(
            agent_name="reporting",
            task_description=task_description,
            context=context,
            technique="chain-of-thought"  # Use CoT for reasoning
        )
        
        logger.info("✨ Using Chain-of-Thought prompt with data context")
        
        # Generate answer using optimized prompt with data
        llm = create_llm_adapter()
        response = llm.generate(optimized_prompt, max_tokens=512)
        
        return response
        
        context = "\n".join(context_parts)
        
        # Generate natural answer using LLM
        llm = create_llm_adapter()
        prompt = f"""Eres un asistente de investigación. Responde esta pregunta usando EXCLUSIVAMENTE los datos proporcionados: {query}

DATOS DISPONIBLES:
{context}

INSTRUCCIONES OBLIGATORIAS:
- USA SOLO los datos proporcionados arriba
- Si preguntan por investigadores, menciona los nombres de autores que aparecen en los papers
- Si preguntan por papers/artículos, menciona los títulos y detalles que están en los datos
- Si preguntan "cuántos", cuenta los documentos encontrados y da el número exacto
- Si preguntan "quiénes", extrae y menciona los nombres de autores de los papers
- Responde en español de forma conversacional y directa
- NO inventes información que no esté en los datos
- Escribe máximo 3 párrafos cortos
- Se específico y usa números/nombres de los datos

Respuesta basada en los datos:"""
        
        response = llm.generate(prompt, max_tokens=600)
        
        # Append top references (only the most relevant ones)
        if documents and len(documents) >= 3:
            response += "\n\n---\n\n**Papers principales mencionados:**\n"
            for i, doc in enumerate(documents[:3], 1):
                title = doc.get("title", "Unknown")
                year = doc.get("year", "")
                doi = doc.get("doi", "")
                response += f"\n• {title} ({year})"
                if doi:
                    response += f" - {doi}"
        
        return response
    
    def _is_conceptual_query(self, query: str) -> bool:
        """
        Determine if query is asking for concept definition/explanation.
        
        Examples:
        - "¿Qué es machine learning?"
        - "Define inteligencia artificial"
        - "Explica qué es deep learning"
        """
        query_lower = query.lower()
        
        conceptual_patterns = [
            "qué es", "que es",
            "define", "defin",
            "explica", "explain",
            "qué significa", "que significa",
            "cómo funciona", "como funciona",
            "what is", "what's",
            "how does", "how do"
        ]
        
        return any(pattern in query_lower for pattern in conceptual_patterns)
    
    def _generate_conceptual_answer(self, query: str, state: AgentState) -> str:
        """
        Generate natural language explanation for conceptual queries.
        Uses retrieved papers as sources but presents in flowing text.
        """
        from marie_agent.adapters.llm_factory import create_llm_adapter
        
        # Get retrieved documents
        evidence_map = state.get("evidence_map", {})
        documents = evidence_map.get("evidence", [])
        
        if not documents:
            # No sources available, use LLM knowledge
            llm = create_llm_adapter()
            prompt = f"""Responde de forma clara y concisa a esta pregunta: {query}

Proporciona una explicación natural y educativa en español."""
            
            response = llm.generate(prompt)
            return response
        
        # Build context from documents
        context_parts = []
        for i, doc in enumerate(documents[:5], 1):
            title = doc.get("title", "Unknown")
            abstract = doc.get("abstract", "")
            year = doc.get("year", "")
            
            if abstract:
                context_parts.append(f"[{i}] {title} ({year}): {abstract[:300]}...")
        
        context = "\n\n".join(context_parts) if context_parts else "No hay contexto disponible."
        
        # Generate natural answer using LLM
        llm = create_llm_adapter()
        prompt = f"""Basándote en los siguientes papers de investigación, responde esta pregunta de forma natural y fluida: {query}

Papers disponibles:
{context}

INSTRUCCIONES:
- Responde en español de forma natural, como si explicaras a un colega
- Usa la información de los papers cuando sea relevante
- Cita los papers entre corchetes [1], [2], etc. cuando uses su información
- NO hagas una lista de papers
- NO uses formato de reporte estructurado
- Genera un texto fluido y coherente de 2-3 párrafos máximo
- Sé conciso y directo

Respuesta:"""
        
        response = llm.generate(prompt, max_tokens=600)
        
        # Append references only if we used papers
        if context_parts:
            response += "\n\n---\n\n**Referencias:**\n"
            for i, doc in enumerate(documents[:5], 1):
                title = doc.get("title", "Unknown")
                year = doc.get("year", "")
                doi = doc.get("doi", "")
                response += f"\n[{i}] {title} ({year})"
                if doi:
                    response += f" - DOI: {doi}"
        
        return response
    
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
    from marie_agent.core.routing import increment_step
    
    agent = ReportingAgent()
    state = agent.generate_report(state)
    return increment_step(state)
