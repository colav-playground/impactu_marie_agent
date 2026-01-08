"""
Human Interaction Manager - Magentic-UI Implementation.

Implements human-in-the-loop patterns:
- Co-Planning: Human collaborates in planning
- Co-Tasking: Human completes specific tasks
- Action Guards: Human approval for critical actions
- Verification: Human validates results
"""

from typing import Dict, Any, Optional, List, Literal
import logging
from datetime import datetime
import uuid

from marie_agent.state import AgentState, HumanInteraction

logger = logging.getLogger(__name__)


class HumanInteractionManager:
    """
    Manages all human-in-the-loop interactions.
    
    Based on Magentic-UI patterns for human-centered AI agents.
    """
    
    def __init__(self, enable_human_interaction: bool = True):
        """
        Initialize human interaction manager.
        
        Args:
            enable_human_interaction: Enable/disable human interaction
        """
        self.enabled = enable_human_interaction
        logger.info(f"Human interaction: {'enabled' if self.enabled else 'disabled'}")
    
    def request_co_planning(
        self,
        state: AgentState,
        proposed_plan: Dict[str, Any],
        reason: str
    ) -> Optional[Dict[str, Any]]:
        """
        Request human collaboration in planning.
        
        Used when:
        - Query is ambiguous
        - Multiple valid interpretations exist
        - High-impact decision needed
        
        Args:
            state: Current agent state
            proposed_plan: AI-generated plan proposal
            reason: Why human input is needed
            
        Returns:
            Modified plan or None if human skipped
        """
        if not self.enabled:
            logger.info("Co-planning requested but human interaction disabled")
            return proposed_plan
        
        interaction_id = f"coplan_{uuid.uuid4().hex[:8]}"
        
        prompt = self._format_co_planning_prompt(proposed_plan, reason)
        
        # Record interaction request
        interaction = HumanInteraction(
            interaction_id=interaction_id,
            type="co_planning",
            prompt=prompt,
            response=None,
            status="pending",
            timestamp=datetime.utcnow().isoformat()
        )
        
        state["human_interactions"].append(interaction)
        state["requires_human_review"] = True
        state["status"] = "waiting_human"
        
        logger.info(f"Co-planning requested: {interaction_id}")
        logger.info(f"Reason: {reason}")
        
        # In CLI mode, this would block and wait for input
        # In API mode, this returns and waits for callback
        return None  # Signals orchestrator to wait
    
    def request_co_tasking(
        self,
        state: AgentState,
        task_description: str,
        context: Dict[str, Any],
        task_type: str
    ) -> Optional[Dict[str, Any]]:
        """
        Request human to complete a specific task.
        
        Used when:
        - Agent cannot complete task autonomously
        - External information needed
        - Specialized knowledge required
        
        Args:
            state: Current agent state
            task_description: What human needs to do
            context: Relevant context for task
            task_type: Type of task (e.g., "entity_disambiguation", "data_collection")
            
        Returns:
            Human-provided result or None
        """
        if not self.enabled:
            logger.info("Co-tasking requested but human interaction disabled")
            return None
        
        interaction_id = f"cotask_{uuid.uuid4().hex[:8]}"
        
        prompt = self._format_co_tasking_prompt(task_description, context, task_type)
        
        interaction = HumanInteraction(
            interaction_id=interaction_id,
            type="co_tasking",
            prompt=prompt,
            response=None,
            status="pending",
            timestamp=datetime.utcnow().isoformat()
        )
        
        state["human_interactions"].append(interaction)
        state["requires_human_review"] = True
        state["status"] = "waiting_human"
        
        logger.info(f"Co-tasking requested: {interaction_id}")
        logger.info(f"Task: {task_description}")
        
        return None
    
    def request_action_guard(
        self,
        state: AgentState,
        action: str,
        impact: str,
        details: Dict[str, Any]
    ) -> bool:
        """
        Request human approval for high-impact action.
        
        Used before:
        - Data modifications
        - External API calls
        - Resource-intensive operations
        - Actions with significant consequences
        
        Args:
            state: Current agent state
            action: Action to be performed
            impact: Why this requires approval
            details: Action details
            
        Returns:
            True if approved, False if rejected
        """
        if not self.enabled:
            logger.info("Action guard requested but human interaction disabled - auto-approving")
            return True
        
        interaction_id = f"guard_{uuid.uuid4().hex[:8]}"
        
        prompt = self._format_action_guard_prompt(action, impact, details)
        
        interaction = HumanInteraction(
            interaction_id=interaction_id,
            type="action_guard",
            prompt=prompt,
            response=None,
            status="pending",
            timestamp=datetime.utcnow().isoformat()
        )
        
        state["human_interactions"].append(interaction)
        state["requires_human_review"] = True
        state["status"] = "waiting_human"
        
        logger.warning(f"Action guard triggered: {interaction_id}")
        logger.warning(f"Action: {action} - Impact: {impact}")
        
        # In production, this would wait for approval
        # For now, log and auto-approve in non-interactive mode
        return False  # Block until approval
    
    def request_verification(
        self,
        state: AgentState,
        result: Dict[str, Any],
        confidence: float,
        concerns: List[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Request human verification of results.
        
        Used when:
        - Confidence below threshold
        - Potential errors detected
        - Critical decisions involved
        - Quality assurance needed
        
        Args:
            state: Current agent state
            result: Result to verify
            confidence: AI confidence score
            concerns: Specific concerns to address
            
        Returns:
            Verified/modified result or None if rejected
        """
        if not self.enabled:
            logger.info("Verification requested but human interaction disabled")
            return result
        
        interaction_id = f"verify_{uuid.uuid4().hex[:8]}"
        
        prompt = self._format_verification_prompt(result, confidence, concerns)
        
        interaction = HumanInteraction(
            interaction_id=interaction_id,
            type="verification",
            prompt=prompt,
            response=None,
            status="pending",
            timestamp=datetime.utcnow().isoformat()
        )
        
        state["human_interactions"].append(interaction)
        state["requires_human_review"] = True
        state["status"] = "waiting_human"
        
        logger.info(f"Verification requested: {interaction_id}")
        logger.info(f"Confidence: {confidence:.2%}, Concerns: {len(concerns)}")
        
        return None
    
    def complete_interaction(
        self,
        state: AgentState,
        interaction_id: str,
        response: Any
    ) -> None:
        """
        Complete a pending human interaction.
        
        Args:
            state: Current agent state
            interaction_id: ID of interaction to complete
            response: Human's response
        """
        for interaction in state["human_interactions"]:
            if interaction["interaction_id"] == interaction_id:
                interaction["response"] = str(response)
                interaction["status"] = "completed"
                logger.info(f"Interaction completed: {interaction_id}")
                
                # Check if all pending interactions are done
                pending = [i for i in state["human_interactions"] if i["status"] == "pending"]
                if not pending:
                    state["requires_human_review"] = False
                    state["status"] = "executing"
                    logger.info("All human interactions completed, resuming execution")
                
                return
        
        logger.warning(f"Interaction not found: {interaction_id}")
    
    def skip_interaction(
        self,
        state: AgentState,
        interaction_id: str,
        reason: str = "Skipped by user"
    ) -> None:
        """
        Skip a pending interaction and continue.
        
        Args:
            state: Current agent state
            interaction_id: ID of interaction to skip
            reason: Why it was skipped
        """
        for interaction in state["human_interactions"]:
            if interaction["interaction_id"] == interaction_id:
                interaction["status"] = "skipped"
                interaction["response"] = reason
                logger.info(f"Interaction skipped: {interaction_id} - {reason}")
                
                # Check if any other interactions pending
                pending = [i for i in state["human_interactions"] if i["status"] == "pending"]
                if not pending:
                    state["requires_human_review"] = False
                    state["status"] = "executing"
                
                return
        
        logger.warning(f"Interaction not found: {interaction_id}")
    
    def _format_co_planning_prompt(
        self,
        proposed_plan: Dict[str, Any],
        reason: str
    ) -> str:
        """Format co-planning prompt for human."""
        steps = proposed_plan.get("steps", [])
        agents = proposed_plan.get("agents_required", [])
        
        prompt = f"""🤔 Co-Planning Request

**Reason:** {reason}

**Proposed Plan:**
Steps: {len(steps)}
{chr(10).join(f"  {i+1}. {step}" for i, step in enumerate(steps))}

Agents: {', '.join(agents)}
Complexity: {proposed_plan.get('estimated_complexity', 'unknown')}

**Options:**
1. Approve plan as-is
2. Modify plan (provide changes)
3. Reject and provide alternative

Your response:"""
        
        return prompt
    
    def _format_co_tasking_prompt(
        self,
        task_description: str,
        context: Dict[str, Any],
        task_type: str
    ) -> str:
        """Format co-tasking prompt for human."""
        prompt = f"""🔧 Co-Tasking Request

**Task Type:** {task_type}
**Description:** {task_description}

**Context:**
{chr(10).join(f"  - {k}: {v}" for k, v in context.items())}

**What to do:**
Please complete this task and provide the result.

Your response:"""
        
        return prompt
    
    def _format_action_guard_prompt(
        self,
        action: str,
        impact: str,
        details: Dict[str, Any]
    ) -> str:
        """Format action guard prompt for human."""
        prompt = f"""⚠️  Action Approval Required

**Action:** {action}
**Impact:** {impact}

**Details:**
{chr(10).join(f"  - {k}: {v}" for k, v in details.items())}

**Approve this action?**
[yes/no]:"""
        
        return prompt
    
    def _format_verification_prompt(
        self,
        result: Dict[str, Any],
        confidence: float,
        concerns: List[str]
    ) -> str:
        """Format verification prompt for human."""
        prompt = f"""✓ Verification Request

**Confidence:** {confidence:.1%}

**Concerns:**
{chr(10).join(f"  - {concern}" for concern in concerns)}

**Result to verify:**
{chr(10).join(f"  {k}: {str(v)[:100]}" for k, v in list(result.items())[:5])}

**Action:**
1. Approve result
2. Modify result (provide corrections)
3. Reject result

Your response:"""
        
        return prompt


# Global instance
_human_manager: Optional[HumanInteractionManager] = None


def get_human_manager(enabled: bool = True) -> HumanInteractionManager:
    """Get or create global human interaction manager."""
    global _human_manager
    if _human_manager is None:
        _human_manager = HumanInteractionManager(enabled)
    return _human_manager
