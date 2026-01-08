"""
Memory System - Stores and retrieves successful execution plans.

Implements plan memory from Magentic-UI paper:
- Save successful plans
- Retrieve similar plans
- Learn from past executions
"""

from typing import List, Dict, Any, Optional
import json
import os
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PlanMemory:
    """
    Stores and retrieves execution plans.
    
    Memory structure:
    - Plans indexed by task description
    - Semantic search for similar tasks
    - Success metrics tracked
    """
    
    def __init__(self, storage_path: str = ".marie_memory"):
        """
        Initialize plan memory.
        
        Args:
            storage_path: Directory to store plans
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.plans_file = self.storage_path / "plans.json"
        
        # Load existing plans
        self.plans: List[Dict[str, Any]] = self._load_plans()
        
        logger.info(f"Plan memory initialized with {len(self.plans)} saved plans")
    
    def save_plan(
        self,
        task: str,
        plan_steps: List[Dict[str, Any]],
        success: bool,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a plan to memory.
        
        Args:
            task: Task description
            plan_steps: List of plan steps
            success: Whether execution was successful
            metadata: Additional metadata (execution time, quality score, etc.)
            
        Returns:
            Plan ID
        """
        plan_id = f"plan_{datetime.utcnow().timestamp()}"
        
        plan_entry = {
            "id": plan_id,
            "task": task,
            "plan_steps": plan_steps,
            "success": success,
            "metadata": metadata or {},
            "created_at": datetime.utcnow().isoformat(),
            "usage_count": 0
        }
        
        self.plans.append(plan_entry)
        self._save_plans()
        
        logger.info(f"Saved plan {plan_id} for task: {task[:50]}...")
        return plan_id
    
    def retrieve_similar_plan(
        self,
        task: str,
        min_similarity: float = 0.7
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve most similar successful plan.
        
        Args:
            task: Task description
            min_similarity: Minimum similarity threshold
            
        Returns:
            Most similar plan or None
        """
        if not self.plans:
            return None
        
        # Filter successful plans
        successful_plans = [p for p in self.plans if p["success"]]
        
        if not successful_plans:
            return None
        
        # Find most similar (simple keyword matching for now)
        best_match = None
        best_score = 0.0
        
        task_lower = task.lower()
        task_words = set(task_lower.split())
        
        for plan in successful_plans:
            plan_task = plan["task"].lower()
            plan_words = set(plan_task.split())
            
            # Jaccard similarity
            intersection = len(task_words & plan_words)
            union = len(task_words | plan_words)
            similarity = intersection / union if union > 0 else 0.0
            
            if similarity > best_score:
                best_score = similarity
                best_match = plan
        
        if best_score >= min_similarity and best_match:
            # Increment usage count
            best_match["usage_count"] += 1
            self._save_plans()
            
            logger.info(f"Retrieved similar plan (similarity={best_score:.2f}) "
                       f"for task: {task[:50]}...")
            return best_match
        
        return None
    
    def get_all_plans(self, successful_only: bool = True) -> List[Dict[str, Any]]:
        """
        Get all stored plans.
        
        Args:
            successful_only: Only return successful plans
            
        Returns:
            List of plans
        """
        if successful_only:
            return [p for p in self.plans if p["success"]]
        return self.plans.copy()
    
    def delete_plan(self, plan_id: str) -> bool:
        """
        Delete a plan from memory.
        
        Args:
            plan_id: ID of plan to delete
            
        Returns:
            True if deleted, False if not found
        """
        original_len = len(self.plans)
        self.plans = [p for p in self.plans if p["id"] != plan_id]
        
        if len(self.plans) < original_len:
            self._save_plans()
            logger.info(f"Deleted plan {plan_id}")
            return True
        
        return False
    
    def clear_all(self) -> None:
        """Clear all plans from memory."""
        self.plans = []
        self._save_plans()
        logger.info("Cleared all plans from memory")
    
    def _load_plans(self) -> List[Dict[str, Any]]:
        """Load plans from disk."""
        if not self.plans_file.exists():
            return []
        
        try:
            with open(self.plans_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading plans: {e}")
            return []
    
    def _save_plans(self) -> None:
        """Save plans to disk."""
        try:
            with open(self.plans_file, 'w') as f:
                json.dump(self.plans, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving plans: {e}")


class EpisodicMemory:
    """
    Stores episodic memories of agent interactions.
    
    Tracks:
    - Query-response pairs
    - User preferences
    - Interaction patterns
    """
    
    def __init__(self, storage_path: str = ".marie_memory"):
        """
        Initialize episodic memory.
        
        Args:
            storage_path: Directory to store memories
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.episodes_file = self.storage_path / "episodes.json"
        
        self.episodes: List[Dict[str, Any]] = self._load_episodes()
        
        logger.info(f"Episodic memory initialized with {len(self.episodes)} episodes")
    
    def save_episode(
        self,
        query: str,
        response: str,
        plan_used: List[Dict[str, Any]],
        success: bool,
        quality_score: Optional[float] = None,
        user_feedback: Optional[str] = None
    ) -> str:
        """
        Save an interaction episode.
        
        Args:
            query: User query
            response: Agent response
            plan_used: Plan that was executed
            success: Whether episode was successful
            quality_score: Quality score if available
            user_feedback: User feedback if provided
            
        Returns:
            Episode ID
        """
        episode_id = f"episode_{datetime.utcnow().timestamp()}"
        
        episode = {
            "id": episode_id,
            "query": query,
            "response": response,
            "plan_used": plan_used,
            "success": success,
            "quality_score": quality_score,
            "user_feedback": user_feedback,
            "created_at": datetime.utcnow().isoformat()
        }
        
        self.episodes.append(episode)
        self._save_episodes()
        
        logger.info(f"Saved episode {episode_id}")
        return episode_id
    
    def get_recent_episodes(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent episodes.
        
        Args:
            n: Number of recent episodes
            
        Returns:
            List of recent episodes
        """
        return self.episodes[-n:]
    
    def get_successful_episodes(self) -> List[Dict[str, Any]]:
        """Get all successful episodes."""
        return [e for e in self.episodes if e["success"]]
    
    def _load_episodes(self) -> List[Dict[str, Any]]:
        """Load episodes from disk."""
        if not self.episodes_file.exists():
            return []
        
        try:
            with open(self.episodes_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading episodes: {e}")
            return []
    
    def _save_episodes(self) -> None:
        """Save episodes to disk."""
        try:
            with open(self.episodes_file, 'w') as f:
                json.dump(self.episodes, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving episodes: {e}")


# Global instances
_plan_memory: Optional[PlanMemory] = None
_episodic_memory: Optional[EpisodicMemory] = None


def get_plan_memory() -> PlanMemory:
    """Get global plan memory instance."""
    global _plan_memory
    if _plan_memory is None:
        _plan_memory = PlanMemory()
    return _plan_memory


def get_episodic_memory() -> EpisodicMemory:
    """Get global episodic memory instance."""
    global _episodic_memory
    if _episodic_memory is None:
        _episodic_memory = EpisodicMemory()
    return _episodic_memory
