"""
Enhanced Action Guard - Validates critical agent actions before execution.

Implements action safety patterns from Magentic-UI:
- Irreversibility detection
- Risk assessment
- User approval workflow
"""

from typing import Dict, Any, Optional, Literal, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ActionProposal:
    """Proposed action for review."""
    action_type: str  # e.g., "data_modification", "external_call", "expensive_operation"
    agent_name: str
    description: str
    parameters: Dict[str, Any]
    irreversibility: Literal["always", "maybe", "never"]
    estimated_cost: Optional[float] = None
    affected_resources: Optional[List[str]] = None


@dataclass
class ActionDecision:
    """Action approval decision."""
    approved: bool
    reason: str
    modified_parameters: Optional[Dict[str, Any]] = None


class ActionGuard:
    """
    Guards critical actions requiring human approval.
    
    Decision flow:
    1. Check irreversibility heuristic
    2. If "always" → request approval
    3. If "maybe" → use LLM judge
    4. If "never" → auto-approve
    """
    
    def __init__(self, llm, auto_approve: bool = False):
        """
        Initialize action guard.
        
        Args:
            llm: LLM for action assessment
            auto_approve: Auto-approve all actions (for testing)
        """
        self.llm = llm
        self.auto_approve = auto_approve
        
        # Action types and their default irreversibility
        self.action_policies = {
            "data_deletion": "always",
            "external_api_call": "maybe",
            "data_modification": "maybe",
            "file_upload": "always",
            "expensive_operation": "maybe",
            "read_operation": "never",
            "search_query": "never"
        }
    
    def check_action(self, proposal: ActionProposal) -> ActionDecision:
        """
        Check if action requires human approval.
        
        Args:
            proposal: Proposed action
            
        Returns:
            Action decision
        """
        if self.auto_approve:
            return ActionDecision(
                approved=True,
                reason="Auto-approval enabled"
            )
        
        # Get irreversibility level
        irreversibility = proposal.irreversibility
        
        # Always irreversible → require approval
        if irreversibility == "always":
            logger.warning(f"Action requires approval: {proposal.description}")
            return self._request_human_approval(proposal)
        
        # Never irreversible → auto-approve
        if irreversibility == "never":
            logger.info(f"Action auto-approved: {proposal.description}")
            return ActionDecision(
                approved=True,
                reason="Action is reversible"
            )
        
        # Maybe irreversible → use LLM judge
        return self._llm_judge_action(proposal)
    
    def _llm_judge_action(self, proposal: ActionProposal) -> ActionDecision:
        """
        Use LLM to judge if action requires approval.
        
        Args:
            proposal: Action proposal
            
        Returns:
            Action decision
        """
        try:
            prompt = f"""
You are an action safety judge. Evaluate if this action requires human approval.

ACTION: {proposal.description}
AGENT: {proposal.agent_name}
TYPE: {proposal.action_type}
PARAMETERS: {proposal.parameters}

Consider:
1. Is this action irreversible?
2. Could it cause unintended consequences?
3. Does it modify important data?
4. Is it a high-cost operation?

Respond with:
REQUIRES_APPROVAL: yes/no
REASON: [brief explanation]
"""
            
            response = self.llm.invoke(prompt)
            
            requires_approval = "yes" in response.lower().split("requires_approval:")[1].split("\n")[0]
            
            if requires_approval:
                return self._request_human_approval(proposal)
            else:
                return ActionDecision(
                    approved=True,
                    reason="LLM judge: action is safe"
                )
        
        except Exception as e:
            logger.error(f"Error in LLM judge: {e}")
            # Fail safe: require approval on error
            return self._request_human_approval(proposal)
    
    def _request_human_approval(self, proposal: ActionProposal) -> ActionDecision:
        """
        Request human approval for action.
        
        Args:
            proposal: Action proposal
            
        Returns:
            Action decision (defaults to rejected in CLI mode)
        """
        logger.warning(f"🚨 ACTION REQUIRES APPROVAL: {proposal.description}")
        logger.warning(f"   Agent: {proposal.agent_name}")
        logger.warning(f"   Type: {proposal.action_type}")
        logger.warning(f"   Irreversibility: {proposal.irreversibility}")
        
        # In CLI mode, default to reject for safety
        # In API mode, this would trigger callback to frontend
        return ActionDecision(
            approved=False,
            reason="Human approval required but not provided"
        )


class InteractionLogger:
    """
    Logs all human-agent interactions for analysis.
    
    Tracks:
    - Co-planning sessions
    - Co-tasking requests
    - Action approvals
    - Verification checks
    """
    
    def __init__(self, log_file: str = ".marie_memory/interactions.log"):
        """
        Initialize interaction logger.
        
        Args:
            log_file: Path to log file
        """
        self.log_file = log_file
    
    def log_interaction(
        self,
        interaction_type: str,
        details: Dict[str, Any],
        user_response: Optional[str] = None
    ) -> None:
        """
        Log an interaction.
        
        Args:
            interaction_type: Type of interaction
            details: Interaction details
            user_response: User's response if provided
        """
        from datetime import datetime
        import json
        
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": interaction_type,
            "details": details,
            "user_response": user_response
        }
        
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Error logging interaction: {e}")


# Global instances
_action_guard: Optional[ActionGuard] = None
_interaction_logger: Optional[InteractionLogger] = None


def get_action_guard(llm, auto_approve: bool = False) -> ActionGuard:
    """Get global action guard instance."""
    global _action_guard
    if _action_guard is None:
        _action_guard = ActionGuard(llm, auto_approve)
    return _action_guard


def get_interaction_logger() -> InteractionLogger:
    """Get global interaction logger instance."""
    global _interaction_logger
    if _interaction_logger is None:
        _interaction_logger = InteractionLogger()
    return _interaction_logger
