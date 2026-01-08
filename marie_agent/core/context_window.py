"""
Context Window Manager - Manages conversation history and prevents overflow.

Maintains cumulative context across agent interactions, summarizes when needed,
and provides relevant context to each agent.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A message in the context window."""
    role: str  # user, agent, system
    content: str
    agent_name: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContextWindow:
    """
    Manages context across agent interactions.
    
    Responsibilities:
    - Store all agent messages and outputs
    - Summarize when context grows too large
    - Provide relevant context to agents
    - Track information gathered
    """
    
    def __init__(self, max_tokens: int = 8000):
        """
        Initialize context window.
        
        Args:
            max_tokens: Maximum tokens before summarization
        """
        self.messages: List[Message] = []
        self.summaries: List[str] = []
        self.max_tokens = max_tokens
        self.current_tokens = 0
        
    def add_message(
        self,
        role: str,
        content: str,
        agent_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a message to context.
        
        Args:
            role: Message role (user, agent, system)
            content: Message content
            agent_name: Name of agent if role is agent
            metadata: Additional metadata
        """
        message = Message(
            role=role,
            content=content,
            agent_name=agent_name,
            metadata=metadata or {}
        )
        
        self.messages.append(message)
        
        # Rough token estimate (1 token ≈ 4 chars)
        self.current_tokens += len(content) // 4
        
        # Summarize if context too large
        if self.current_tokens > self.max_tokens:
            self._summarize_old_messages()
    
    def get_context_for_agent(
        self,
        agent_name: str,
        include_summaries: bool = True
    ) -> str:
        """
        Get relevant context for an agent.
        
        Args:
            agent_name: Name of agent requesting context
            include_summaries: Include previous summaries
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        # Add summaries if requested
        if include_summaries and self.summaries:
            context_parts.append("=== Previous Context Summary ===")
            context_parts.extend(self.summaries)
            context_parts.append("")
        
        # Add recent messages
        context_parts.append("=== Recent Messages ===")
        for msg in self.messages[-20:]:  # Last 20 messages
            if msg.role == "user":
                context_parts.append(f"User: {msg.content}")
            elif msg.role == "agent":
                context_parts.append(f"{msg.agent_name}: {msg.content}")
            elif msg.role == "system":
                context_parts.append(f"System: {msg.content}")
        
        return "\n".join(context_parts)
    
    def get_full_history(self) -> List[Message]:
        """Get all messages in history."""
        return self.messages.copy()
    
    def get_progress_summary(self) -> str:
        """
        Generate a summary of progress so far.
        
        Returns:
            Summary of what has been accomplished
        """
        if not self.messages:
            return "No progress yet."
        
        # Find agent messages
        agent_messages = [m for m in self.messages if m.role == "agent"]
        
        if not agent_messages:
            return "Task initiated, awaiting agent actions."
        
        # Summarize agent actions
        summary_parts = [f"Progress: {len(agent_messages)} agent actions taken:"]
        
        for i, msg in enumerate(agent_messages[-5:], 1):  # Last 5
            summary_parts.append(f"{i}. {msg.agent_name}: {msg.content[:100]}...")
        
        return "\n".join(summary_parts)
    
    def _summarize_old_messages(self) -> None:
        """
        Summarize older messages to reduce context size.
        
        Keeps recent messages, summarizes older ones.
        """
        # Keep last 10 messages
        messages_to_keep = 10
        messages_to_summarize = self.messages[:-messages_to_keep]
        
        if not messages_to_summarize:
            return
        
        # Create summary
        summary_parts = [
            f"=== Summary of {len(messages_to_summarize)} earlier messages ==="
        ]
        
        # Group by agent
        agent_actions: Dict[str, List[str]] = {}
        for msg in messages_to_summarize:
            if msg.role == "agent" and msg.agent_name:
                if msg.agent_name not in agent_actions:
                    agent_actions[msg.agent_name] = []
                agent_actions[msg.agent_name].append(msg.content[:100])
        
        for agent, actions in agent_actions.items():
            summary_parts.append(f"{agent}: {len(actions)} actions")
            summary_parts.append(f"  - {actions[0]}...")
        
        summary = "\n".join(summary_parts)
        self.summaries.append(summary)
        
        # Keep only recent messages
        self.messages = self.messages[-messages_to_keep:]
        
        # Recalculate tokens
        self.current_tokens = sum(len(m.content) // 4 for m in self.messages)
        
        logger.info(f"Summarized {len(messages_to_summarize)} messages, "
                   f"reduced to {self.current_tokens} tokens")
    
    def clear(self) -> None:
        """Clear all context."""
        self.messages.clear()
        self.summaries.clear()
        self.current_tokens = 0
