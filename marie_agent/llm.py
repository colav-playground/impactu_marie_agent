"""
LLM Integration for MARIE agents.

Provides LLM capabilities for query understanding, entity extraction, and reasoning.
"""

from typing import Dict, Any, List, Optional
import logging

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from marie_agent.config import config

logger = logging.getLogger(__name__)


class MarieLLM:
    """
    LLM interface for MARIE agents.
    
    Uses Claude for reasoning, planning, and understanding.
    """
    
    def __init__(self):
        """Initialize LLM client."""
        if not config.llm.api_key:
            logger.warning("No API key configured. LLM features will use fallbacks.")
            self.llm = None
        else:
            self.llm = ChatAnthropic(
                model=config.llm.model,
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_tokens,
                api_key=config.llm.api_key
            )
            logger.info(f"LLM initialized: {config.llm.model}")
    
    @property
    def available(self) -> bool:
        """Check if LLM is available."""
        return self.llm is not None
    
    def parse_query(self, query: str) -> Dict[str, Any]:
        """
        Parse user query to extract intent, entities, and filters.
        
        Args:
            query: User's natural language query
            
        Returns:
            Parsed query structure with intent, entities, filters
        """
        if not self.available:
            logger.debug("LLM not available, using rule-based parsing")
            return self._parse_query_fallback(query)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a scientometric query analyzer for CRIS (Current Research Information System).

Extract the following from the user query:
1. Intent: What type of analysis (e.g., "top_papers", "author_productivity", "collaboration_network")
2. Entities: Names of institutions, authors, research groups
3. Filters: Year ranges, document types, specific criteria
4. Metrics: What metrics to compute (citations, h-index, publications count)
5. Limit: How many results to return

Respond ONLY with valid JSON following this schema:
{{
  "intent": "string",
  "entities": {{
    "institutions": ["string"],
    "authors": ["string"],
    "groups": ["string"]
  }},
  "filters": {{
    "year_start": "integer or null",
    "year_end": "integer or null",
    "document_types": ["string"]
  }},
  "metrics": ["string"],
  "limit": "integer",
  "complexity": "simple|medium|complex"
}}"""),
            ("human", "{query}")
        ])
        
        parser = JsonOutputParser()
        chain = prompt | self.llm | parser
        
        try:
            result = chain.invoke({"query": query})
            logger.info(f"Query parsed: intent={result.get('intent')}")
            return result
        except Exception as e:
            logger.error(f"Error parsing query: {e}")
            return self._parse_query_fallback(query)
    
    def _parse_query_fallback(self, query: str) -> Dict[str, Any]:
        """Rule-based fallback for query parsing."""
        query_lower = query.lower()
        
        # Detect intent
        intent = "search"
        if "top" in query_lower or "most cited" in query_lower:
            intent = "top_papers"
        elif "author" in query_lower and ("productivity" in query_lower or "publications" in query_lower):
            intent = "author_productivity"
        
        # Extract limit
        limit = 5
        for num in [3, 5, 10, 20, 50]:
            if str(num) in query:
                limit = num
                break
        
        return {
            "intent": intent,
            "entities": {"institutions": [], "authors": [], "groups": []},
            "filters": {},
            "metrics": ["citations"],
            "limit": limit,
            "complexity": "simple"
        }
    
    def extract_entities(self, query: str, entity_type: str) -> List[Dict[str, Any]]:
        """
        Extract specific entity type from query.
        
        Args:
            query: User query
            entity_type: Type of entity (institution, author, group)
            
        Returns:
            List of extracted entities with confidence
        """
        if not self.available:
            return []
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""Extract {entity_type} names from the query.

For each {entity_type}, provide:
- name: The extracted name
- aliases: Possible alternative names or abbreviations
- confidence: Your confidence (0.0-1.0)

Respond ONLY with valid JSON:
{{
  "entities": [
    {{"name": "string", "aliases": ["string"], "confidence": 0.0}}
  ]
}}"""),
            ("human", "{query}")
        ])
        
        parser = JsonOutputParser()
        chain = prompt | self.llm | parser
        
        try:
            result = chain.invoke({"query": query})
            entities = result.get("entities", [])
            logger.info(f"Extracted {len(entities)} {entity_type}(s)")
            return entities
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []
    
    def generate_answer(
        self,
        query: str,
        context: Dict[str, Any],
        metrics: Dict[str, Any]
    ) -> str:
        """
        Generate natural language answer from context and metrics.
        
        Args:
            query: Original user query
            context: Retrieved context and evidence
            metrics: Computed metrics
            
        Returns:
            Natural language answer
        """
        if not self.available:
            logger.debug("LLM not available, using template-based answer")
            return None  # Let reporting agent handle it
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are MARIE, a scientometric analysis assistant for CRIS.

Generate a clear, evidence-based answer to the user's query.

Rules:
1. Base ALL claims on provided evidence
2. Cite specific numbers and sources
3. Be concise but informative
4. State limitations if data is incomplete
5. Use markdown formatting
6. Include specific paper titles, authors, years, citations

Format your answer with:
- Clear title
- Numbered list of top items (if applicable)
- Summary statistics
- Brief methodology note"""),
            ("human", """User Query: {query}

Context: {context}

Metrics: {metrics}

Generate a comprehensive answer:""")
        ])
        
        chain = prompt | self.llm
        
        try:
            result = chain.invoke({
                "query": query,
                "context": str(context)[:2000],  # Limit context
                "metrics": str(metrics)[:1000]
            })
            answer = result.content
            logger.info(f"Generated answer: {len(answer)} chars")
            return answer
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "Error generating answer. Please try again."
    
    def create_plan(self, query: str, parsed_query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create execution plan for the query.
        
        Args:
            query: Original query
            parsed_query: Parsed query structure
        if not self.available:
            return self._create_default_plan()
        
            
        Returns:
            Execution plan with steps and agents
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a scientometric analysis planner.

Given a parsed query, create an execution plan specifying:
1. Which agents to execute (entity_resolution, retrieval, validation, metrics, citations, reporting)
2. Order of execution
3. Whether human input is needed
4. Complexity estimate

Available agents:
- entity_resolution: Disambiguate authors/institutions
- retrieval: Search MongoDB and OpenSearch
- validation: Check data consistency
- metrics: Compute scientometric indicators
- citations: Build evidence map
- reporting: Generate final report

Respond with JSON:
{{
  "steps": ["agent_name"],
  "agents_required": ["agent_name"],
  "requires_human_input": boolean,
  "estimated_complexity": "simple|medium|complex",
  "reasoning": "string"
}}"""),
            ("human", """Query: {query}
Parsed: {parsed_query}

Create execution plan:""")
        ])
        
        parser = JsonOutputParser()
        chain = prompt | self.llm | parser
        
        try:
            result = chain.invoke({
                "query": query,
                "parsed_query": str(parsed_query)
            })
            logger.info(f"Created plan: {len(result.get('steps', []))} steps")
            return result
        except Exception as e:
            logger.error(f"Error creating plan: {e}")
            return self._create_default_plan()
    
    def _create_default_plan(self) -> Dict[str, Any]:
        """Create default execution plan."""
        return {
            "steps": [
                "Resolve entities",
                "Retrieve evidence",
                "Validate data",
                "Compute metrics",
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
            "requires_human_input": False,
            "estimated_complexity": "medium",
            "reasoning": "Default plan"
        }
    
    def assess_confidence(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess confidence in the results.
        
        if not self.available:
            return self._assess_confidence_fallback(state)
        
        Args:
            state: Current agent state
            
        Returns:
            Confidence assessment with score and explanation
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Assess the confidence in these scientometric results.

Consider:
1. Data completeness (how much data was found)
2. Entity resolution quality
3. Evidence strength
4. Metric reliability

Provide:
- confidence_score: 0.0-1.0
- confidence_level: "low|medium|high"
- reasoning: Why this confidence level
- limitations: What data or resolution issues exist

Respond with JSON:
{{
  "confidence_score": 0.0,
  "confidence_level": "string",
  "reasoning": "string",
  "limitations": ["string"]
}}"""),
            ("human", """Evidence count: {evidence_count}
Entities resolved: {entities_resolved}
Documents retrieved: {docs_count}
Citations: {citations_count}

Assess confidence:""")
        ])
        
        parser = JsonOutputParser()
        chain = prompt | self.llm | parser
        
        try:
            result = chain.invoke({
                "evidence_count": len(state.get("evidence_map", {})),
                "entities_resolved": len(state.get("entities_resolved", {}).get("institutions", [])),
                "docs_count": len(state.get("retrieved_data", [])),
                "citations_count": len(state.get("citations", []))
            })
            logger.info(f"Confidence assessed: {result.get('confidence_level')}")
            return result
        except Exception as e:
            logger.error(f"Error assessing confidence: {e}")
            return self._assess_confidence_fallback(state)
    
    def _assess_confidence_fallback(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Rule-based confidence assessment."""
        evidence_count = len(state.get("evidence_map", {}))
        docs_count = len(state.get("retrieved_data", []))
        citations_count = len(state.get("citations", []))
        
        # Simple heuristic
        score = 0.0
        if evidence_count > 0:
            score += 0.3
        if docs_count >= 5:
            score += 0.4
        if citations_count >= 3:
            score += 0.3
        
        level = "low"
        if score >= 0.7:
            level = "high"
        elif score >= 0.4:
            level = "medium"
        
        return {
            "confidence_score": score,
            "confidence_level": level,
            "reasoning": f"Based on {evidence_count} evidence sources, {docs_count} documents, {citations_count} citations",
            "limitations": ["Rule-based assessment without LLM analysis"]
        }


# Global LLM instance
_llm_instance: Optional[MarieLLM] = None


def get_llm() -> MarieLLM:
    """Get or create global LLM instance."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = MarieLLM()
    return _llm_instance
