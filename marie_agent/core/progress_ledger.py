"""
Progress Ledger - Tracks execution progress and determines next actions.

Maintains state for each plan step: completion status, replan needs,
and instructions for agents.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class StepCompletion:
    """Status of step completion."""
    answer: bool  # True if complete
    reason: str   # Why it is/isn't complete


@dataclass
class ReplanNeed:
    """Whether replanning is needed."""
    answer: bool  # True if replan needed
    reason: str   # Why replan is/isn't needed


@dataclass
class Instruction:
    """Instruction for agent."""
    answer: str       # Detailed instruction
    agent_name: str   # Agent to execute


@dataclass
class ProgressLedger:
    """
    Progress ledger for a plan step.
    
    Tracks:
    - step_complete: Is this step done?
    - replan: Do we need to change the plan?
    - instruction: What should the agent do?
    - progress_summary: What have we accomplished?
    """
    step_complete: StepCompletion
    replan: ReplanNeed
    instruction: Instruction
    progress_summary: str
    
    def __str__(self) -> str:
        """String representation."""
        return (
            f"ProgressLedger(\n"
            f"  complete={self.step_complete.answer}, "
            f"  replan={self.replan.answer},\n"
            f"  agent={self.instruction.agent_name},\n"
            f"  summary={self.progress_summary[:100]}...\n"
            f")"
        )


class ProgressTracker:
    """
    Generates and manages progress ledgers with variable storage.
    
    Uses LLM to assess progress and determine next actions.
    Stores task outputs as variables for referencing in subsequent tasks.
    """
    
    def __init__(self, llm):
        """
        Initialize progress tracker with variable storage (Magentic + ReWOO).
        
        Args:
            llm: LLM for generating ledgers
        """
        self.llm = llm
        self.variable_store: Dict[str, Any] = {}  # Store task outputs ($E1, $E2, etc.)
        self.execution_history: list = []  # Track all executed tasks
        
        logger.info("ProgressTracker initialized with variable storage")
    
    def store_variable(self, var_name: str, value: Any, task_id: str = "") -> None:
        """
        Store a variable from task execution (ReWOO-style).
        
        Args:
            var_name: Variable name (e.g., 'researchers', 'h_indices')
            value: Variable value
            task_id: Task ID that produced this (e.g., 'E1')
        """
        # Remove $ prefix if present
        var_name = var_name.lstrip('$')
        
        self.variable_store[var_name] = value
        logger.info(f"✓ Variable stored: ${var_name} (from {task_id or 'unknown'})")
        
        # Track in history
        self.execution_history.append({
            'task_id': task_id,
            'variable': var_name,
            'value_type': type(value).__name__,
            'value_size': len(value) if isinstance(value, (list, dict, str)) else 1
        })
    
    def get_variable(self, var_name: str) -> Any:
        """
        Retrieve a variable value.
        
        Args:
            var_name: Variable name (with or without $ prefix)
            
        Returns:
            Variable value or None if not found
        """
        var_name = var_name.lstrip('$')
        value = self.variable_store.get(var_name)
        
        if value is None:
            logger.warning(f"Variable ${var_name} not found in store")
        
        return value
    
    def resolve_variables(self, text: str) -> str:
        """
        Replace $variable references with actual values in text.
        
        Args:
            text: Text containing $var references
            
        Returns:
            Text with variables resolved
        """
        import re
        
        # Find all $var patterns
        var_pattern = r'\$(\w+)'
        matches = re.findall(var_pattern, text)
        
        resolved_text = text
        for var_name in matches:
            value = self.get_variable(var_name)
            if value is not None:
                # Convert value to string representation
                if isinstance(value, (list, dict)):
                    value_str = f"<{type(value).__name__} with {len(value)} items>"
                else:
                    value_str = str(value)
                
                resolved_text = resolved_text.replace(f"${var_name}", value_str)
                logger.debug(f"Resolved ${var_name} in text")
        
        return resolved_text
    
    def get_available_variables(self) -> Dict[str, str]:
        """
        Get list of available variables with their types.
        
        Returns:
            Dict of variable names to type descriptions
        """
        return {
            f"${name}": type(value).__name__
            for name, value in self.variable_store.items()
        }
    
    def clear_variables(self) -> None:
        """Clear all stored variables (for new query)."""
        self.variable_store.clear()
        self.execution_history.clear()
        logger.info("Variable store cleared")
    
    def generate_ledger(
        self,
        query: str,
        plan: list,
        current_step_index: int,
        context: str,
        evidence_map: Dict[str, Any]
    ) -> ProgressLedger:
        """
        Generate progress ledger for current step.
        
        Args:
            query: Original user query
            plan: List of plan steps
            current_step_index: Index of current step
            context: Accumulated context
            evidence_map: Evidence gathered so far
            
        Returns:
            Progress ledger with status and instructions
        """
        logger.info(f"Generating ledger for step {current_step_index + 1}/{len(plan)}")
        
        try:
            current_step = plan[current_step_index]
            
            # Build prompt
            prompt = self._build_ledger_prompt(
                query=query,
                plan=plan,
                current_step_index=current_step_index,
                context=context,
                evidence_map=evidence_map
            )
            
            # Get LLM response
            response = self.llm.generate(prompt)
            
            # Parse response
            ledger = self._parse_ledger_response(response, current_step)
            
            logger.info(f"Ledger: complete={ledger.step_complete.answer}, "
                       f"replan={ledger.replan.answer}")
            
            return ledger
            
        except Exception as e:
            logger.error(f"Error generating ledger: {e}", exc_info=True)
            
            # Default ledger: continue with current step
            return ProgressLedger(
                step_complete=StepCompletion(
                    answer=False,
                    reason="Continuing execution"
                ),
                replan=ReplanNeed(
                    answer=False,
                    reason="No replan needed"
                ),
                instruction=Instruction(
                    answer=current_step.get("details", "Execute step"),
                    agent_name=current_step.get("agent_name", "reporting")
                ),
                progress_summary="Continuing with plan"
            )
    
    def _build_ledger_prompt(
        self,
        query: str,
        plan: list,
        current_step_index: int,
        context: str,
        evidence_map: Dict[str, Any]
    ) -> str:
        """Build prompt for ledger generation."""
        
        current_step = plan[current_step_index]
        
        # Count evidence
        evidence_count = sum(len(v) if isinstance(v, list) else 1 
                           for v in evidence_map.values())
        
        prompt_parts = [
            "You are an orchestrator evaluating progress on a multi-step plan.",
            "",
            f"USER QUERY: {query}",
            "",
            "PLAN:",
        ]
        
        for i, step in enumerate(plan):
            status = "✓ DONE" if i < current_step_index else "→ CURRENT" if i == current_step_index else "PENDING"
            prompt_parts.append(f"{i+1}. [{status}] {step.get('agent_name')}: {step.get('title')}")
        
        prompt_parts.extend([
            "",
            f"CURRENT STEP: {current_step.get('title')}",
            f"AGENT: {current_step.get('agent_name')}",
            "",
            f"CONTEXT SO FAR:\n{context[:1000]}...",
            "",
            f"EVIDENCE GATHERED: {evidence_count} documents",
            "",
            "Evaluate and respond with:",
            "",
            "STEP_COMPLETE: [yes/no]",
            "REASON: [why step is/isn't complete]",
            "",
            "REPLAN_NEEDED: [yes/no]",
            "REPLAN_REASON: [why replan is/isn't needed]",
            "",
            "INSTRUCTION: [detailed instruction for agent]",
            "AGENT_NAME: [agent to execute]",
            "",
            "PROGRESS_SUMMARY: [summary of accomplishments so far]"
        ])
        
        return "\n".join(prompt_parts)
    
    def _parse_ledger_response(
        self,
        response: str,
        current_step: dict
    ) -> ProgressLedger:
        """Parse LLM response into progress ledger."""
        
        lines = response.split("\n")
        
        # Defaults
        step_complete = False
        complete_reason = "Continuing execution"
        replan = False
        replan_reason = "No replan needed"
        instruction = current_step.get("details", "Execute step")
        agent_name = current_step.get("agent_name", "reporting")
        progress_summary = "Making progress on task"
        
        # Parse response
        for i, line in enumerate(lines):
            line = line.strip()
            
            if "STEP_COMPLETE:" in line.upper():
                step_complete = "yes" in line.lower()
            
            elif "REASON:" in line.upper() and i > 0:
                if "STEP_COMPLETE" in lines[i-1].upper():
                    complete_reason = line.split(":", 1)[1].strip()
            
            elif "REPLAN_NEEDED:" in line.upper():
                replan = "yes" in line.lower()
            
            elif "REPLAN_REASON:" in line.upper():
                replan_reason = line.split(":", 1)[1].strip()
            
            elif "INSTRUCTION:" in line.upper():
                instruction = line.split(":", 1)[1].strip()
            
            elif "AGENT_NAME:" in line.upper():
                agent_name = line.split(":", 1)[1].strip()
            
            elif "PROGRESS_SUMMARY:" in line.upper():
                progress_summary = line.split(":", 1)[1].strip()
        
        return ProgressLedger(
            step_complete=StepCompletion(
                answer=step_complete,
                reason=complete_reason
            ),
            replan=ReplanNeed(
                answer=replan,
                reason=replan_reason
            ),
            instruction=Instruction(
                answer=instruction,
                agent_name=agent_name
            ),
            progress_summary=progress_summary
        )
