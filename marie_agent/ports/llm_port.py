"""
LLM Port - Interface for language model interactions.

Defines the contract for LLM services in hexagonal architecture.
"""

from typing import Dict, Any, List, Protocol
from abc import ABC, abstractmethod


class LLMPort(ABC):
    """
    Port (interface) for Language Model services.
    
    This defines the contract that any LLM adapter must implement.
    """
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if LLM service is available."""
        pass
    
    @abstractmethod
    def parse_query(self, query: str) -> Dict[str, Any]:
        """
        Parse user query to extract intent, entities, and filters.
        
        Args:
            query: User's natural language query
            
        Returns:
            Parsed query structure
        """
        pass
    
    @abstractmethod
    def extract_entities(self, query: str, entity_type: str) -> List[Dict[str, Any]]:
        """
        Extract specific entity type from query.
        
        Args:
            query: User query
            entity_type: Type of entity to extract
            
        Returns:
            List of extracted entities
        """
        pass
    
    @abstractmethod
    def create_plan(self, query: str, parsed_query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create execution plan for query.
        
        Args:
            query: Original query
            parsed_query: Parsed query structure
            
        Returns:
            Execution plan
        """
        pass
    
    @abstractmethod
    def assess_confidence(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess confidence in results.
        
        Args:
            state: Current agent state
            
        Returns:
            Confidence assessment
        """
        pass
    
    @abstractmethod
    def think(self, prompt: str, context: str = "") -> str:
        """
        Generate reasoning/thinking response.
        
        Args:
            prompt: The thinking prompt
            context: Optional context
            
        Returns:
            Reasoning output
        """
        pass
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """
        Generate text completion.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        pass
