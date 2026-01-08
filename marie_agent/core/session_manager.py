"""
Session Manager - Manages multiple concurrent agent sessions.

Enables multitasking from Magentic-UI:
- Multiple parallel sessions
- Session state management
- Background execution
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import logging
import uuid

from marie_agent.state import AgentState, create_initial_state

logger = logging.getLogger(__name__)


@dataclass
class SessionInfo:
    """Information about an active session."""
    session_id: str
    status: str  # running, paused, completed, failed
    created_at: str
    updated_at: str
    user_query: str
    requires_user_input: bool
    progress_percent: float


class SessionManager:
    """
    Manages multiple concurrent agent sessions.
    
    Features:
    - Create and track multiple sessions
    - Pause/resume sessions
    - Session status monitoring
    - Background execution
    """
    
    def __init__(self):
        """Initialize session manager."""
        self.sessions: Dict[str, AgentState] = {}
        self.max_concurrent_sessions = 10
        
        logger.info("Session manager initialized")
    
    def create_session(self, user_query: str) -> str:
        """
        Create a new session.
        
        Args:
            user_query: User's query
            
        Returns:
            Session ID
        """
        # Check limit
        if len(self.sessions) >= self.max_concurrent_sessions:
            self._cleanup_completed_sessions()
            
            if len(self.sessions) >= self.max_concurrent_sessions:
                raise Exception(f"Max concurrent sessions ({self.max_concurrent_sessions}) reached")
        
        # Create session
        session_id = str(uuid.uuid4())
        state = create_initial_state(user_query, session_id)
        
        self.sessions[session_id] = state
        
        logger.info(f"Created session {session_id}: {user_query[:50]}...")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[AgentState]:
        """
        Get session state.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session state or None
        """
        return self.sessions.get(session_id)
    
    def update_session(self, session_id: str, state: AgentState) -> None:
        """
        Update session state.
        
        Args:
            session_id: Session ID
            state: Updated state
        """
        if session_id in self.sessions:
            state["updated_at"] = datetime.utcnow().isoformat()
            self.sessions[session_id] = state
        else:
            logger.warning(f"Attempted to update non-existent session {session_id}")
    
    def pause_session(self, session_id: str) -> bool:
        """
        Pause a running session.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if paused, False otherwise
        """
        state = self.sessions.get(session_id)
        if state and state["status"] == "executing":
            state["status"] = "waiting_human"
            state["updated_at"] = datetime.utcnow().isoformat()
            logger.info(f"Paused session {session_id}")
            return True
        return False
    
    def resume_session(self, session_id: str) -> bool:
        """
        Resume a paused session.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if resumed, False otherwise
        """
        state = self.sessions.get(session_id)
        if state and state["status"] == "waiting_human":
            state["status"] = "executing"
            state["updated_at"] = datetime.utcnow().isoformat()
            logger.info(f"Resumed session {session_id}")
            return True
        return False
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if deleted, False if not found
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Deleted session {session_id}")
            return True
        return False
    
    def list_sessions(self) -> List[SessionInfo]:
        """
        List all active sessions.
        
        Returns:
            List of session info
        """
        sessions = []
        
        for session_id, state in self.sessions.items():
            # Calculate progress
            plan = state.get("plan") or {}  # Handle None
            total_steps = len(plan.get("steps", []))
            current_step = state.get("current_step", 0)
            
            progress = (current_step / total_steps * 100) if total_steps > 0 else 0
            
            # Check if user input needed
            requires_input = (
                state.get("status") == "waiting_human" or
                state.get("requires_human_review", False)
            )
            
            info = SessionInfo(
                session_id=session_id,
                status=state.get("status", "unknown"),
                created_at=state.get("created_at", ""),
                updated_at=state.get("updated_at", ""),
                user_query=state.get("user_query", ""),
                requires_user_input=requires_input,
                progress_percent=progress
            )
            
            sessions.append(info)
        
        return sessions
    
    def get_session_count(self) -> Dict[str, int]:
        """
        Get session counts by status.
        
        Returns:
            Dict with counts per status
        """
        counts = {
            "total": len(self.sessions),
            "running": 0,
            "paused": 0,
            "completed": 0,
            "failed": 0
        }
        
        for state in self.sessions.values():
            status = state["status"]
            if status == "executing":
                counts["running"] += 1
            elif status == "waiting_human":
                counts["paused"] += 1
            elif status == "completed":
                counts["completed"] += 1
            elif status == "failed":
                counts["failed"] += 1
        
        return counts
    
    def _cleanup_completed_sessions(self) -> int:
        """
        Remove completed/failed sessions.
        
        Returns:
            Number of sessions removed
        """
        to_remove = [
            sid for sid, state in self.sessions.items()
            if state["status"] in ["completed", "failed"]
        ]
        
        for session_id in to_remove:
            del self.sessions[session_id]
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} completed sessions")
        
        return len(to_remove)


# Global instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get global session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
