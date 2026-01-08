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
            # Use Prompt Engineer + Language Detector for NER
            from marie_agent.agents.prompt_engineer import get_prompt_engineer
            from marie_agent.adapters.llm_factory import get_llm_adapter
            from marie_agent.services.language_detector import get_language_detector
            
            prompt_engineer = get_prompt_engineer()
            llm = get_llm_adapter()
            lang_detector = get_language_detector()
            
            # Detect language for better extraction
            detected_lang = lang_detector.detect(query)
            
            # Prepare context with language info
            context = {
                "query": query,
                "language": detected_lang,
                "task": "extract_entities"
            }
            
            # Build few-shot NER prompt
            ner_examples = [
                {
                    "input": "papers from Universidad de Antioquia",
                    "output": "INSTITUTION: Universidad de Antioquia"
                },
                {
                    "input": "publications by John Smith",
                    "output": "PERSON: John Smith"
                },
                {
                    "input": "research at MIT",
                    "output": "INSTITUTION: MIT"
                }
            ]
            
            extraction_prompt = prompt_engineer.build_prompt(
                agent_name="entity_resolution",
                task_description="Extract institutions, authors, and topics from query",
                context=context,
                technique="few-shot",
                examples=ner_examples
            )
            
            # Extract entities using LLM
            llm_response = llm.generate(extraction_prompt, max_tokens=200)
            logger.debug(f"NER response: {llm_response}")
            
            # Parse response and resolve against database
            entities = {
                "institutions": self._resolve_institutions(query, llm_response),
                "authors": self._resolve_authors(query, llm_response),
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
    
    def _resolve_institutions(self, query: str, llm_extracted: str = "") -> List[Dict[str, Any]]:
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
    
    def _resolve_authors(self, query: str, llm_extracted: str = "") -> List[Dict[str, Any]]:
        """
        Resolve author names using LLM extraction.
        
        Args:
            query: User query
            llm_extracted: LLM extracted entities
            
        Returns:
            List of resolved authors with metadata
        """
        # Use LLM extraction combined with keyword search
        author_keywords = ["author:", "researcher:", "by ", "autor:"]
        
        for keyword in author_keywords:
            if keyword in query.lower():
                parts = query.lower().split(keyword)
                if len(parts) > 1:
                    author_name = " ".join(parts[1].split()[0:3])
                    return [{
                        "name": author_name,
                        "id": f"author_{author_name.replace(' ', '_')}",
                        "confidence": 0.7
                    }]
        
        return []
    
    def _resolve_with_tool(self, name: str, entity_type: str) -> Dict[str, Any]:
        """
        Resolve entity using tools.py with retry logic.
        
        Args:
            name: Entity name
            entity_type: Type (institution/author)
            
        Returns:
            Entity metadata or empty dict
        """
        max_retries = 2
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                from marie_agent.tools import resolve_entity
                
                logger.debug(f"Resolving {entity_type} '{name}' (attempt {retry_count + 1})")
                
                formatted, metadata = resolve_entity.invoke({
                    "name": name,
                    "entity_type": entity_type
                })
                
                if metadata:
                    logger.info(f"✓ Resolved {entity_type}: {name}")
                    return metadata
                else:
                    logger.warning(f"Tool returned no metadata for {name}")
                    return {}
                    
            except Exception as e:
                retry_count += 1
                logger.error(f"Tool error for {name} (attempt {retry_count}): {e}")
                
                if retry_count > max_retries:
                    return {}
                    
                import time
                time.sleep(0.5 * retry_count)
        
        return {}
    
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
    from marie_agent.core.routing import increment_step
    
    agent = EntityResolutionAgent()
    state = agent.resolve(state)
    return increment_step(state)
