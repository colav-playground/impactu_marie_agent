"""
Entity Resolution Agent - Disambiguate entities.

Responsible for:
- Resolving author/researcher names
- Disambiguating institutions
- Fuzzy matching with confidence scores
- Requesting human input when confidence is low
"""

from typing import Dict, Any, List
import logging
from datetime import datetime

from pymongo import MongoClient

from marie_agent.state import AgentState, add_audit_event, create_task, add_task, start_task, complete_task, fail_task
from marie_agent.config import config

logger = logging.getLogger(__name__)


class EntityResolutionAgent:
    """
    Agent responsible for entity disambiguation.
    
    Resolves ambiguous references to authors, institutions, etc.
    """
    
    def __init__(self):
        """Initialize entity resolution agent."""
        self.mongo_client = MongoClient(config.mongodb.uri)
        self.mongo_db = self.mongo_client[config.mongodb.database]
        
        logger.info("Entity resolution agent initialized")
    
    def resolve(self, state: AgentState) -> AgentState:
        """
        Resolve entities in the query.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with resolved entities
        """
        query = state["user_query"]
        
        # Create and start task
        task = create_task(
            agent="entity_resolution",
            input_data={"query": query}
        )
        add_task(state, task)
        start_task(state, task["task_id"])
        
        logger.info(f"Resolving entities for: {query}")
        
        try:
            # TODO: Use NER/LLM to extract entities from query
            # For now, do simple keyword matching
            
            entities = {
                "institutions": self._resolve_institutions(query),
                "authors": self._resolve_authors(query),
                "groups": []
            }
            
            state["entities_resolved"] = entities
            
            # Complete task
            complete_task(
                state,
                task["task_id"],
                output=entities,
                confidence=0.9 if entities["institutions"] or entities["authors"] else 0.5
            )
            
            add_audit_event(state, "entity_resolution_completed", {
                "institutions_found": len(entities["institutions"]),
                "authors_found": len(entities["authors"])
            })
            
            state["next_agent"] = "retrieval"
            
            logger.info(f"Resolved {len(entities['institutions'])} institutions")
            
        except Exception as e:
            logger.error(f"Error during entity resolution: {e}", exc_info=True)
            fail_task(state, task["task_id"], str(e))
            state["error"] = str(e)
            state["next_agent"] = None
            state["status"] = "failed"
        
        return state
    
    def _resolve_institutions(self, query: str) -> List[Dict[str, Any]]:
        """
        Resolve institution names from query.
        
        Args:
            query: User query
            
        Returns:
            List of resolved institutions with metadata
        """
        # Simple keyword search in affiliations collection
        keywords = self._extract_institution_keywords(query)
        
        if not keywords:
            return []
        
        try:
            # Search affiliations collection
            results = self.mongo_db.affiliations.find(
                {
                    "$or": [
                        {"names.name": {"$regex": kw, "$options": "i"}}
                        for kw in keywords
                    ]
                },
                limit=5
            )
            
            institutions = []
            for doc in results:
                institutions.append({
                    "id": str(doc["_id"]),
                    "name": doc["names"][0]["name"] if doc.get("names") else "Unknown",
                    "confidence": 0.8,  # TODO: Calculate actual confidence
                    "type": doc.get("types", [{}])[0].get("type", "institution")
                })
            
            return institutions
            
        except Exception as e:
            logger.error(f"Institution resolution error: {e}")
            return []
    
    def _resolve_authors(self, query: str) -> List[Dict[str, Any]]:
        """
        Resolve author names from query.
        
        Args:
            query: User query
            
        Returns:
            List of resolved authors with metadata
        """
        # TODO: Implement author name resolution
        return []
    
    def _extract_institution_keywords(self, query: str) -> List[str]:
        """
        Extract institution name keywords from query.
        
        Args:
            query: User query
            
        Returns:
            List of keywords
        """
        # Simple heuristic - look for common institution patterns
        keywords = []
        
        # Known institution names
        known_institutions = [
            "Universidad de Antioquia",
            "Universidad Nacional",
            "Universidad de los Andes",
            "Universidad del Valle",
            "UdeA"
        ]
        
        query_lower = query.lower()
        for inst in known_institutions:
            if inst.lower() in query_lower:
                keywords.append(inst)
        
        return keywords


def entity_resolution_agent_node(state: AgentState) -> AgentState:
    """
    Entity resolution agent node function for LangGraph.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with resolved entities
    """
    agent = EntityResolutionAgent()
    return agent.resolve(state)
