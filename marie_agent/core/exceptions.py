"""
Custom exceptions for MARIE agent system.

Provides structured error handling across all components.
"""


class MarieException(Exception):
    """Base exception for all MARIE errors."""
    
    def __init__(self, message: str, details: dict = None):
        """
        Initialize MARIE exception.
        
        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(MarieException):
    """Configuration or environment error."""
    pass


class AgentExecutionError(MarieException):
    """Error during agent execution."""
    
    def __init__(self, agent_name: str, message: str, details: dict = None):
        """
        Initialize agent execution error.
        
        Args:
            agent_name: Name of the agent that failed
            message: Error message
            details: Additional error details
        """
        super().__init__(message, details)
        self.agent_name = agent_name


class StateValidationError(MarieException):
    """Invalid state encountered."""
    pass


class OpenSearchError(MarieException):
    """OpenSearch operation error."""
    pass


class MongoDBError(MarieException):
    """MongoDB operation error."""
    pass


class LLMError(MarieException):
    """LLM provider error."""
    
    def __init__(self, provider: str, message: str, details: dict = None):
        """
        Initialize LLM error.
        
        Args:
            provider: LLM provider name (ollama, vllm, etc)
            message: Error message
            details: Additional error details
        """
        super().__init__(message, details)
        self.provider = provider


class MemoryError(MarieException):
    """Memory system error."""
    pass


class QueryGenerationError(AgentExecutionError):
    """Error generating OpenSearch query."""
    pass


class PromptGenerationError(AgentExecutionError):
    """Error generating prompt."""
    pass


class EntityResolutionError(AgentExecutionError):
    """Error resolving entities."""
    pass


class RetrievalError(AgentExecutionError):
    """Error retrieving documents."""
    pass


class MetricsComputationError(AgentExecutionError):
    """Error computing metrics."""
    pass
