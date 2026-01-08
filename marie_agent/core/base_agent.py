"""
Base classes and protocols for MARIE agents.

Provides common functionality and interfaces for all agents.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Protocol
import logging
from datetime import datetime

from marie_agent.state import AgentState
from marie_agent.config import config


class AgentProtocol(Protocol):
    """Protocol defining the interface for all agents."""
    
    def execute(self, state: AgentState) -> AgentState:
        """
        Execute agent logic.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state
        """
        ...


class BaseAgent(ABC):
    """
    Base class for all MARIE agents.
    
    Provides common functionality:
    - Logging setup
    - Error handling patterns
    - State validation
    - Execution timing
    """
    
    def __init__(self, agent_name: str):
        """
        Initialize base agent.
        
        Args:
            agent_name: Name of the agent (for logging)
        """
        self.agent_name = agent_name
        self.logger = logging.getLogger(f"marie_agent.agents.{agent_name}")
        self.logger.info(f"{agent_name} agent initialized")
    
    @abstractmethod
    def execute(self, state: AgentState) -> AgentState:
        """
        Execute agent logic. Must be implemented by subclasses.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state
        """
        pass
    
    def _validate_state(self, state: AgentState) -> None:
        """
        Validate state before execution.
        
        Args:
            state: State to validate
            
        Raises:
            ValueError: If state is invalid
        """
        if not isinstance(state, dict):
            raise ValueError(f"State must be a dict, got {type(state)}")
        
        if "query" not in state:
            raise ValueError("State must contain 'query'")
    
    def _log_execution_start(self, state: AgentState) -> float:
        """
        Log execution start and return start time.
        
        Args:
            state: Current state
            
        Returns:
            Start timestamp
        """
        self.logger.info(f"🚀 Starting {self.agent_name}")
        return datetime.now().timestamp()
    
    def _log_execution_end(self, start_time: float, success: bool = True) -> None:
        """
        Log execution end with timing.
        
        Args:
            start_time: Execution start timestamp
            success: Whether execution succeeded
        """
        duration = datetime.now().timestamp() - start_time
        status = "✓" if success else "✗"
        self.logger.info(f"{status} {self.agent_name} completed in {duration:.2f}s")
    
    def safe_execute(self, state: AgentState) -> AgentState:
        """
        Execute with error handling and logging.
        
        Args:
            state: Current state
            
        Returns:
            Updated state
        """
        start_time = None
        try:
            # Validate
            self._validate_state(state)
            
            # Log start
            start_time = self._log_execution_start(state)
            
            # Execute
            result = self.execute(state)
            
            # Log success
            self._log_execution_end(start_time, success=True)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in {self.agent_name}: {e}", exc_info=True)
            
            if start_time:
                self._log_execution_end(start_time, success=False)
            
            # Add error to state
            state["error"] = str(e)
            state["failed_agent"] = self.agent_name
            
            return state


class MemoryEnabledAgent(BaseAgent):
    """
    Base class for agents that use memory/logging systems.
    
    Extends BaseAgent with memory initialization patterns.
    """
    
    def __init__(self, agent_name: str, enable_memory: bool = True):
        """
        Initialize memory-enabled agent.
        
        Args:
            agent_name: Name of the agent
            enable_memory: Whether to enable memory system
        """
        super().__init__(agent_name)
        self.enable_memory = enable_memory
        
        if enable_memory:
            self._setup_memory()
    
    @abstractmethod
    def _setup_memory(self) -> None:
        """Setup memory system. Must be implemented by subclasses."""
        pass
    
    def _log_to_memory(self, data: Dict[str, Any]) -> None:
        """
        Log data to memory system if enabled.
        
        Args:
            data: Data to log
        """
        if not self.enable_memory:
            return
        
        try:
            self._save_to_memory(data)
        except Exception as e:
            self.logger.warning(f"Failed to log to memory: {e}")
    
    @abstractmethod
    def _save_to_memory(self, data: Dict[str, Any]) -> None:
        """Save data to memory. Must be implemented by subclasses."""
        pass


class ReflexionAgent(MemoryEnabledAgent):
    """
    Base class for agents using reflexion pattern.
    
    Provides iterative refinement with self-evaluation.
    """
    
    def __init__(
        self,
        agent_name: str,
        max_iterations: int,
        enable_memory: bool = True
    ):
        """
        Initialize reflexion agent.
        
        Args:
            agent_name: Name of the agent
            max_iterations: Maximum refinement attempts
            enable_memory: Whether to enable memory
        """
        super().__init__(agent_name, enable_memory)
        self.max_iterations = max_iterations
    
    @abstractmethod
    def _generate_attempt(
        self,
        state: AgentState,
        iteration: int,
        reflection: Optional[str] = None
    ) -> Any:
        """
        Generate an attempt. Must be implemented by subclasses.
        
        Args:
            state: Current state
            iteration: Current iteration number
            reflection: Reflection from previous iteration
            
        Returns:
            Generated result
        """
        pass
    
    @abstractmethod
    def _evaluate_result(self, result: Any, state: AgentState) -> bool:
        """
        Evaluate if result is satisfactory. Must be implemented by subclasses.
        
        Args:
            result: Result to evaluate
            state: Current state
            
        Returns:
            True if result is good enough
        """
        pass
    
    @abstractmethod
    def _reflect(
        self,
        result: Any,
        state: AgentState,
        iteration: int
    ) -> str:
        """
        Reflect on failed result. Must be implemented by subclasses.
        
        Args:
            result: Failed result
            state: Current state
            iteration: Current iteration
            
        Returns:
            Reflection string for next iteration
        """
        pass
    
    def execute_with_reflexion(self, state: AgentState) -> tuple[Any, int]:
        """
        Execute with reflexion pattern.
        
        Args:
            state: Current state
            
        Returns:
            Tuple of (best_result, iterations_used)
        """
        best_result = None
        reflection = None
        
        for iteration in range(1, self.max_iterations + 1):
            self.logger.debug(f"Reflexion iteration {iteration}/{self.max_iterations}")
            
            # Generate attempt
            result = self._generate_attempt(state, iteration, reflection)
            
            # Evaluate
            is_good = self._evaluate_result(result, state)
            
            if is_good or result is not None:
                best_result = result
                
                if is_good:
                    self.logger.debug(f"✓ Good result on iteration {iteration}")
                    return best_result, iteration
            
            # Reflect for next iteration
            if iteration < self.max_iterations:
                reflection = self._reflect(result, state, iteration)
                self.logger.debug(f"Reflection: {reflection[:100]}...")
        
        return best_result, self.max_iterations
