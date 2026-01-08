"""
Memory Manager - Long-term memory and context preservation.

Implements:
- Conversation history tracking
- Long-term memory persistence
- Context preservation across sessions
- Learning from feedback
"""

from typing import Dict, Any, List, Optional
import logging
import json
from datetime import datetime
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


class ConversationTurn:
    """Single turn in a conversation."""
    
    def __init__(
        self,
        turn_id: str,
        user_query: str,
        agent_response: str,
        timestamp: str,
        confidence: float,
        evidence_count: int,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.turn_id = turn_id
        self.user_query = user_query
        self.agent_response = agent_response
        self.timestamp = timestamp
        self.confidence = confidence
        self.evidence_count = evidence_count
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "turn_id": self.turn_id,
            "user_query": self.user_query,
            "agent_response": self.agent_response,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "evidence_count": self.evidence_count,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationTurn":
        """Create from dictionary."""
        return cls(
            turn_id=data["turn_id"],
            user_query=data["user_query"],
            agent_response=data["agent_response"],
            timestamp=data["timestamp"],
            confidence=data["confidence"],
            evidence_count=data["evidence_count"],
            metadata=data.get("metadata", {})
        )


class ConversationSession:
    """Multi-turn conversation session."""
    
    def __init__(self, session_id: str, user_id: Optional[str] = None):
        self.session_id = session_id
        self.user_id = user_id
        self.turns: List[ConversationTurn] = []
        self.started_at = datetime.utcnow().isoformat()
        self.context: Dict[str, Any] = {}
        self.topics: List[str] = []
        
    def add_turn(self, turn: ConversationTurn) -> None:
        """Add turn to conversation with enhanced logging."""
        self.turns.append(turn)
        logger.info(f"✓ Turn {turn.turn_id} added to session {self.session_id} "
                   f"(confidence: {turn.confidence:.2f}, evidence: {turn.evidence_count})")
        
        # Log memory stats
        total_turns = len(self.turns)
        avg_confidence = sum(t.confidence for t in self.turns) / total_turns if total_turns > 0 else 0
        logger.debug(f"Session {self.session_id}: {total_turns} turns, avg confidence: {avg_confidence:.2f}")
    
    def get_recent_context(self, n: int = 3) -> List[ConversationTurn]:
        """Get n most recent turns for context with logging."""
        recent = self.turns[-n:] if len(self.turns) > n else self.turns
        logger.debug(f"Retrieved {len(recent)} recent turns for context")
        return recent
    
    def update_context(self, key: str, value: Any) -> None:
        """Update session context."""
        self.context[key] = value
    
    def add_topic(self, topic: str) -> None:
        """Add topic to session."""
        if topic not in self.topics:
            self.topics.append(topic)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "started_at": self.started_at,
            "turns": [turn.to_dict() for turn in self.turns],
            "context": self.context,
            "topics": self.topics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationSession":
        """Create from dictionary."""
        session = cls(
            session_id=data["session_id"],
            user_id=data.get("user_id")
        )
        session.started_at = data["started_at"]
        session.turns = [ConversationTurn.from_dict(t) for t in data["turns"]]
        session.context = data.get("context", {})
        session.topics = data.get("topics", [])
        return session


class MemoryStore:
    """Persistent storage for long-term memory."""
    
    def __init__(self, storage_path: str = ".marie_memory"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.sessions_path = self.storage_path / "sessions"
        self.sessions_path.mkdir(exist_ok=True)
        self.knowledge_path = self.storage_path / "knowledge.json"
        self.feedback_path = self.storage_path / "feedback.json"
        
        logger.info(f"Memory store initialized: {self.storage_path}")
    
    def save_session(self, session: ConversationSession) -> None:
        """Save conversation session."""
        try:
            session_file = self.sessions_path / f"{session.session_id}.json"
            with open(session_file, 'w') as f:
                json.dump(session.to_dict(), f, indent=2)
            logger.debug(f"Session saved: {session.session_id}")
        except Exception as e:
            logger.error(f"Error saving session: {e}")
    
    def load_session(self, session_id: str) -> Optional[ConversationSession]:
        """Load conversation session."""
        try:
            session_file = self.sessions_path / f"{session_id}.json"
            if session_file.exists():
                with open(session_file, 'r') as f:
                    data = json.load(f)
                return ConversationSession.from_dict(data)
        except Exception as e:
            logger.error(f"Error loading session: {e}")
        return None
    
    def list_sessions(self, user_id: Optional[str] = None) -> List[str]:
        """List all session IDs, optionally filtered by user."""
        sessions = []
        for session_file in self.sessions_path.glob("*.json"):
            if user_id:
                try:
                    with open(session_file, 'r') as f:
                        data = json.load(f)
                    if data.get("user_id") == user_id:
                        sessions.append(session_file.stem)
                except Exception:
                    continue
            else:
                sessions.append(session_file.stem)
        return sessions
    
    def save_knowledge(self, knowledge: Dict[str, Any]) -> None:
        """Save learned knowledge."""
        try:
            existing = self.load_knowledge()
            existing.update(knowledge)
            with open(self.knowledge_path, 'w') as f:
                json.dump(existing, f, indent=2)
            logger.debug(f"Knowledge saved: {len(knowledge)} items")
        except Exception as e:
            logger.error(f"Error saving knowledge: {e}")
    
    def load_knowledge(self) -> Dict[str, Any]:
        """Load learned knowledge."""
        try:
            if self.knowledge_path.exists():
                with open(self.knowledge_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading knowledge: {e}")
        return {}
    
    def save_feedback(self, feedback: Dict[str, Any]) -> None:
        """Save user feedback."""
        try:
            feedbacks = self.load_all_feedback()
            feedbacks.append({
                **feedback,
                "timestamp": datetime.utcnow().isoformat()
            })
            with open(self.feedback_path, 'w') as f:
                json.dump(feedbacks, f, indent=2)
            logger.debug("Feedback saved")
        except Exception as e:
            logger.error(f"Error saving feedback: {e}")
    
    def load_all_feedback(self) -> List[Dict[str, Any]]:
        """Load all feedback."""
        try:
            if self.feedback_path.exists():
                with open(self.feedback_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading feedback: {e}")
        return []


class MemoryManager:
    """
    Manages long-term memory and conversation history.
    
    Enables:
    - Multi-turn conversations with context
    - Learning from past interactions
    - User preferences and patterns
    - Feedback incorporation
    """
    
    def __init__(self, storage_path: str = ".marie_memory"):
        self.store = MemoryStore(storage_path)
        self.active_sessions: Dict[str, ConversationSession] = {}
        self.knowledge = self.store.load_knowledge()
        logger.info("Memory manager initialized")
    
    def create_session(
        self,
        session_id: str,
        user_id: Optional[str] = None
    ) -> ConversationSession:
        """Create new conversation session."""
        session = ConversationSession(session_id, user_id)
        self.active_sessions[session_id] = session
        logger.info(f"Session created: {session_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get active or load persisted session."""
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
        
        # Try loading from storage
        session = self.store.load_session(session_id)
        if session:
            self.active_sessions[session_id] = session
            logger.info(f"Session loaded: {session_id}")
        return session
    
    def record_turn(
        self,
        session_id: str,
        turn_id: str,
        user_query: str,
        agent_response: str,
        confidence: float,
        evidence_count: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record conversation turn."""
        session = self.get_session(session_id)
        if not session:
            session = self.create_session(session_id)
        
        turn = ConversationTurn(
            turn_id=turn_id,
            user_query=user_query,
            agent_response=agent_response,
            timestamp=datetime.utcnow().isoformat(),
            confidence=confidence,
            evidence_count=evidence_count,
            metadata=metadata
        )
        
        session.add_turn(turn)
        logger.info(f"Turn recorded: {turn_id} in session {session_id}")
    
    def get_conversation_context(
        self,
        session_id: str,
        turns: int = 3
    ) -> List[Dict[str, str]]:
        """Get recent conversation context for LLM."""
        session = self.get_session(session_id)
        if not session:
            return []
        
        recent_turns = session.get_recent_context(turns)
        context = []
        
        for turn in recent_turns:
            context.extend([
                {"role": "user", "content": turn.user_query},
                {"role": "assistant", "content": turn.agent_response}
            ])
        
        return context
    
    def save_session(self, session_id: str) -> None:
        """Persist session to storage."""
        session = self.active_sessions.get(session_id)
        if session:
            self.store.save_session(session)
            logger.info(f"Session persisted: {session_id}")
    
    def close_session(self, session_id: str) -> None:
        """Close and persist session."""
        self.save_session(session_id)
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.info(f"Session closed: {session_id}")
    
    def record_feedback(
        self,
        session_id: str,
        turn_id: str,
        rating: int,
        comment: Optional[str] = None,
        corrections: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record user feedback on a response."""
        feedback = {
            "session_id": session_id,
            "turn_id": turn_id,
            "rating": rating,
            "comment": comment,
            "corrections": corrections
        }
        
        self.store.save_feedback(feedback)
        
        # Learn from corrections
        if corrections:
            self._learn_from_corrections(corrections)
        
        logger.info(f"Feedback recorded: rating={rating}, turn={turn_id}")
    
    def _learn_from_corrections(self, corrections: Dict[str, Any]) -> None:
        """Learn from user corrections."""
        # Simple learning: Store corrections as knowledge
        for key, value in corrections.items():
            knowledge_key = f"correction_{key}"
            if knowledge_key in self.knowledge:
                if isinstance(self.knowledge[knowledge_key], list):
                    self.knowledge[knowledge_key].append(value)
                else:
                    self.knowledge[knowledge_key] = [self.knowledge[knowledge_key], value]
            else:
                self.knowledge[knowledge_key] = value
        
        self.store.save_knowledge(self.knowledge)
        logger.debug("Learned from corrections")
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get learned preferences for user."""
        user_sessions = self.store.list_sessions(user_id)
        
        # Analyze past sessions for patterns
        preferences = {
            "session_count": len(user_sessions),
            "topics": [],
            "avg_confidence": 0.0,
            "common_queries": []
        }
        
        # Load and analyze sessions
        all_topics = []
        confidences = []
        
        for session_id in user_sessions[-10:]:  # Last 10 sessions
            session = self.store.load_session(session_id)
            if session:
                all_topics.extend(session.topics)
                confidences.extend([t.confidence for t in session.turns])
        
        if all_topics:
            from collections import Counter
            preferences["topics"] = [t for t, _ in Counter(all_topics).most_common(5)]
        
        if confidences:
            preferences["avg_confidence"] = sum(confidences) / len(confidences)
        
        return preferences
    
    def get_knowledge(self, key: str) -> Optional[Any]:
        """Retrieve learned knowledge."""
        return self.knowledge.get(key)
    
    def update_knowledge(self, key: str, value: Any) -> None:
        """Update knowledge base."""
        self.knowledge[key] = value
        self.store.save_knowledge({key: value})
        logger.debug(f"Knowledge updated: {key}")


# Global instance
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager(storage_path: str = ".marie_memory") -> MemoryManager:
    """Get or create global memory manager."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager(storage_path)
    return _memory_manager
