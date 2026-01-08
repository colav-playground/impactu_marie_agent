"""
State management for MARIE multi-agent system.

Defines the shared state structure used across all agents in the system.
Based on Magentic Orchestration pattern with Ledger as source of truth.
"""

from typing import TypedDict, List, Dict, Any, Optional, Literal
from datetime import datetime


class Evidence(TypedDict):
    """Evidence item linking claims to sources."""
    source_type: Literal["mongodb", "opensearch", "file"]
    source_id: str
    collection: Optional[str]
    reference: str
    content: str
    confidence: float
    timestamp: str


class Task(TypedDict):
    """Task representation in the ledger."""
    task_id: str
    agent: str
    status: Literal["pending", "in_progress", "completed", "failed", "needs_human"]
    input: Dict[str, Any]
    output: Optional[Dict[str, Any]]
    evidence: List[Evidence]
    confidence: Optional[float]
    error: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]


class HumanInteraction(TypedDict):
    """Human-in-the-loop interaction record."""
    interaction_id: str
    type: Literal["co_planning", "co_tasking", "action_guard", "verification"]
    prompt: str
    response: Optional[str]
    status: Literal["pending", "completed", "skipped"]
    timestamp: str


class Plan(TypedDict):
    """Resolution plan for the query."""
    steps: List[str]
    agents_required: List[str]
    estimated_complexity: Literal["simple", "medium", "complex"]
    requires_human_input: bool
    filters: Dict[str, Any]


class AgentState(TypedDict):
    """
    Shared state for the multi-agent system.
    
    This is the single source of truth, passed between all agents.
    Based on Magentic Orchestration and Magentic-UI principles.
    """
    # Request metadata
    request_id: str
    user_query: str
    created_at: str
    updated_at: str
    
    # Planning
    plan: Optional[Plan]
    current_step: int
    parsed_query: Optional[Dict[str, Any]]  # Parsed query with intent
    needs_rag: Optional[bool]  # Whether RAG is needed
    
    # Tasks and execution
    tasks: List[Task]
    
    # Evidence collection
    evidence_map: Dict[str, List[Evidence]]  # claim_id -> evidence list
    
    # Human interaction
    human_interactions: List[HumanInteraction]
    requires_human_review: bool
    
    # Results
    entities_resolved: Dict[str, Any]  # entity_type -> resolved entities
    retrieved_data: List[Dict[str, Any]]
    computed_metrics: Dict[str, Any]
    citations: List[Dict[str, Any]]
    
    # Final output
    final_answer: Optional[str]
    report: Optional[str]
    confidence_score: Optional[float]
    confidence_assessment: Optional[Dict[str, Any]]
    
    # Magentic additions
    quality_report: Optional[Dict[str, Any]]  # Quality evaluation
    refinement_count: Optional[int]  # Number of plan refinements
    replan_reason: Optional[str]  # Why replanning was triggered
    
    # Audit trail
    audit_log: List[Dict[str, Any]]
    
    # Control flow
    next_agent: Optional[str]
    status: Literal["planning", "executing", "replanning", "ready_for_quality_check", "waiting_human", "completed", "failed"]
    error: Optional[str]


def create_initial_state(user_query: str, request_id: str) -> AgentState:
    """
    Create initial state for a new query.
    
    Args:
        user_query: User's question
        request_id: Unique request identifier
        
    Returns:
        Initial agent state
    """
    now = datetime.utcnow().isoformat()
    
    return AgentState(
        request_id=request_id,
        user_query=user_query,
        created_at=now,
        updated_at=now,
        plan=None,
        current_step=0,
        parsed_query=None,
        needs_rag=None,
        tasks=[],
        evidence_map={},
        human_interactions=[],
        requires_human_review=False,
        entities_resolved={},
        retrieved_data=[],
        computed_metrics={},
        citations=[],
        final_answer=None,
        report=None,
        confidence_score=None,
        confidence_assessment=None,
        quality_report=None,
        refinement_count=0,
        replan_reason=None,
        audit_log=[
            {
                "timestamp": now,
                "event": "request_created",
                "details": {"query": user_query}
            }
        ],
        next_agent=None,
        status="planning",
        error=None
    )


def add_audit_event(state: AgentState, event: str, details: Dict[str, Any]) -> None:
    """
    Add an audit event to the state.
    
    Args:
        state: Current state
        event: Event name
        details: Event details
    """
    state["audit_log"].append({
        "timestamp": datetime.utcnow().isoformat(),
        "event": event,
        "details": details
    })
    state["updated_at"] = datetime.utcnow().isoformat()


def create_task(
    agent: str,
    input_data: Dict[str, Any],
    task_id: Optional[str] = None
) -> Task:
    """
    Create a new task for an agent.
    
    Args:
        agent: Agent name
        input_data: Input parameters for the task
        task_id: Optional task ID, auto-generated if not provided
        
    Returns:
        New task object
    """
    import uuid
    
    return Task(
        task_id=task_id or f"{agent}_{uuid.uuid4().hex[:8]}",
        agent=agent,
        status="pending",
        input=input_data,
        output=None,
        evidence=[],
        confidence=None,
        error=None,
        started_at=None,
        completed_at=None
    )


def add_task(state: AgentState, task: Task) -> None:
    """
    Add a new task to the state.
    
    Args:
        state: Current state
        task: Task to add
    """
    state["tasks"].append(task)
    add_audit_event(state, "task_created", {
        "task_id": task["task_id"],
        "agent": task["agent"]
    })


def start_task(state: AgentState, task_id: str) -> None:
    """
    Mark a task as started.
    
    Args:
        state: Current state
        task_id: Task ID to start
    """
    for task in state["tasks"]:
        if task["task_id"] == task_id:
            task["status"] = "in_progress"
            task["started_at"] = datetime.utcnow().isoformat()
            add_audit_event(state, "task_started", {"task_id": task_id, "agent": task["agent"]})
            return


def complete_task(
    state: AgentState,
    task_id: str,
    output: Dict[str, Any],
    evidence: Optional[List[Evidence]] = None,
    confidence: Optional[float] = None
) -> None:
    """
    Mark a task as completed with results.
    
    Args:
        state: Current state
        task_id: Task ID to complete
        output: Task output/results
        evidence: Optional evidence collected
        confidence: Optional confidence score
    """
    for task in state["tasks"]:
        if task["task_id"] == task_id:
            task["status"] = "completed"
            task["output"] = output
            task["evidence"] = evidence or []
            task["confidence"] = confidence
            task["completed_at"] = datetime.utcnow().isoformat()
            
            add_audit_event(state, "task_completed", {
                "task_id": task_id,
                "agent": task["agent"],
                "confidence": confidence
            })
            return


def fail_task(state: AgentState, task_id: str, error: str) -> None:
    """
    Mark a task as failed with error.
    
    Args:
        state: Current state
        task_id: Task ID to fail
        error: Error message
    """
    for task in state["tasks"]:
        if task["task_id"] == task_id:
            task["status"] = "failed"
            task["error"] = error
            task["completed_at"] = datetime.utcnow().isoformat()
            
            add_audit_event(state, "task_failed", {
                "task_id": task_id,
                "agent": task["agent"],
                "error": error
            })
            return
