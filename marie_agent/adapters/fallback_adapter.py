"""
Fallback Adapter - Rule-based LLM port implementation.

Used when no LLM is available. Provides basic functionality using heuristics.
"""

from typing import Dict, Any, List
import logging
import re

from marie_agent.ports.llm_port import LLMPort

logger = logging.getLogger(__name__)


class FallbackAdapter(LLMPort):
    """
    Fallback adapter using rule-based methods.
    
    Used when vLLM or API-based LLMs are not available.
    """
    
    def __init__(self):
        """Initialize fallback adapter."""
        logger.info("Fallback adapter initialized (rule-based, no LLM)")
        self._available = False  # No actual LLM available
    
    def is_available(self) -> bool:
        """Check if LLM is available (always False for fallback)."""
        return self._available
    
    def parse_query(self, query: str) -> Dict[str, Any]:
        """Parse query using rule-based heuristics."""
        query_lower = query.lower()
        
        # Detect intent
        intent = "search"
        if "top" in query_lower or "most cited" in query_lower:
            intent = "top_papers"
        elif "author" in query_lower and ("productivity" in query_lower or "publications" in query_lower):
            intent = "author_productivity"
        elif "collaboration" in query_lower or "network" in query_lower:
            intent = "collaboration_network"
        
        # Extract limit
        limit = 5
        numbers = re.findall(r'\b(\d+)\b', query)
        for num in numbers:
            n = int(num)
            if 1 <= n <= 100:
                limit = n
                break
        
        # Extract year ranges
        years = re.findall(r'\b(20\d{2})\b', query)
        year_start = None
        year_end = None
        if len(years) == 1:
            year_start = int(years[0])
        elif len(years) >= 2:
            year_start = min(int(y) for y in years)
            year_end = max(int(y) for y in years)
        
        result = {
            "intent": intent,
            "entities": {"institutions": [], "authors": [], "groups": []},
            "filters": {
                "year_start": year_start,
                "year_end": year_end,
                "document_types": []
            },
            "metrics": ["citations"],
            "limit": limit,
            "complexity": "simple"
        }
        
        logger.debug(f"Parsed query: intent={intent}, limit={limit}")
        return result
    
    def extract_entities(self, query: str, entity_type: str) -> List[Dict[str, Any]]:
        """Extract entities using pattern matching."""
        # Basic pattern matching for common institutions
        entities = []
        
        if entity_type == "institution":
            patterns = [
                r"Universidad de Antioquia|UdeA",
                r"Universidad Nacional",
                r"Universidad de los Andes",
                r"Universidad del Valle"
            ]
            
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    match = re.search(pattern, query, re.IGNORECASE)
                    entities.append({
                        "name": match.group(0),
                        "aliases": [],
                        "confidence": 0.7
                    })
        
        return entities
    
    def create_plan(self, query: str, parsed_query: Dict[str, Any]) -> Dict[str, Any]:
        """Create execution plan using rule-based logic."""
        complexity = parsed_query.get("complexity", "simple")
        intent = parsed_query.get("intent", "search")
        
        # Basic plan based on intent
        if intent == "top_papers":
            steps = [
                "Resolve institution entities",
                "Retrieve papers from database",
                "Compute citation metrics",
                "Build citations",
                "Generate report"
            ]
            agents = ["entity_resolution", "retrieval", "metrics", "citations", "reporting"]
        else:
            steps = [
                "Resolve entities",
                "Retrieve evidence",
                "Validate data",
                "Compute metrics",
                "Build citations",
                "Generate report"
            ]
            agents = [
                "entity_resolution",
                "retrieval",
                "validation",
                "metrics",
                "citations",
                "reporting"
            ]
        
        return {
            "steps": steps,
            "agents_required": agents,
            "requires_human_input": False,
            "estimated_complexity": complexity,
            "reasoning": f"Rule-based plan for {intent} query"
        }
    
    def assess_confidence(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Assess confidence using simple heuristics."""
        evidence_count = len(state.get("evidence_map", {}))
        docs_count = len(state.get("retrieved_data", []))
        citations_count = len(state.get("citations", []))
        entities_count = len(state.get("entities_resolved", {}).get("institutions", []))
        
        # Simple scoring
        score = 0.0
        factors = []
        
        if evidence_count > 0:
            score += 0.25
            factors.append(f"{evidence_count} evidence sources")
        
        if docs_count >= 10:
            score += 0.35
            factors.append(f"{docs_count} documents")
        elif docs_count >= 5:
            score += 0.25
            factors.append(f"{docs_count} documents")
        elif docs_count > 0:
            score += 0.15
            factors.append(f"{docs_count} documents")
        
        if citations_count >= 5:
            score += 0.25
            factors.append(f"{citations_count} citations")
        elif citations_count > 0:
            score += 0.15
            factors.append(f"{citations_count} citations")
        
        if entities_count > 0:
            score += 0.15
            factors.append(f"{entities_count} entities resolved")
        
        # Cap score at 1.0
        score = min(score, 1.0)
        
        # Determine level
        level = "low"
        if score >= 0.75:
            level = "high"
        elif score >= 0.5:
            level = "medium"
        
        reasoning = "Based on " + ", ".join(factors) if factors else "Limited data available"
        
        return {
            "confidence_score": min(score, 1.0),
            "confidence_level": level,
            "reasoning": reasoning,
            "limitations": [
                "Rule-based assessment (no LLM reasoning)",
                "Limited to available indexed data"
            ]
        }
    
    def think(self, prompt: str, context: str = "") -> str:
        """Thinking not available in fallback mode."""
        logger.debug("Think called but not available in fallback mode")
        return ""
    
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generation not available in fallback mode."""
        logger.debug("Generate called but not available in fallback mode")
        return ""
