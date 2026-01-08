"""
Enhanced state management with Pydantic models.

Provides type-safe state with validation, default values, and serialization.
"""

from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_HUMAN = "needs_human"


class InteractionType(str, Enum):
    """Human interaction types."""
    CO_PLANNING = "co_planning"
    CO_TASKING = "co_tasking"
    ACTION_GUARD = "action_guard"
    VERIFICATION = "verification"


class InteractionStatus(str, Enum):
    """Interaction status."""
    PENDING = "pending"
    COMPLETED = "completed"
    SKIPPED = "skipped"


class SourceType(str, Enum):
    """Evidence source types."""
    MONGODB = "mongodb"
    OPENSEARCH = "opensearch"
    FILE = "file"


class Complexity(str, Enum):
    """Plan complexity levels."""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


class Evidence(BaseModel):
    """Evidence item linking claims to sources."""
    source_type: SourceType
    source_id: str
    collection: Optional[str] = None
    reference: str
    content: str
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0-1")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    @validator('confidence')
    def validate_confidence(cls, v):
        """Ensure confidence is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v
    
    class Config:
        use_enum_values = True


class Task(BaseModel):
    """Task representation in the ledger."""
    task_id: str
    agent: str
    status: TaskStatus = TaskStatus.PENDING
    input: Dict[str, Any] = Field(default_factory=dict)
    output: Optional[Dict[str, Any]] = None
    evidence: List[Evidence] = Field(default_factory=list)
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    
    def start(self) -> None:
        """Mark task as started."""
        self.status = TaskStatus.IN_PROGRESS
        self.started_at = datetime.now().isoformat()
    
    def complete(self, output: Dict[str, Any], confidence: Optional[float] = None) -> None:
        """Mark task as completed."""
        self.status = TaskStatus.COMPLETED
        self.output = output
        self.confidence = confidence
        self.completed_at = datetime.now().isoformat()
        
        if self.started_at:
            start = datetime.fromisoformat(self.started_at)
            end = datetime.fromisoformat(self.completed_at)
            self.duration_seconds = (end - start).total_seconds()
    
    def fail(self, error: str) -> None:
        """Mark task as failed."""
        self.status = TaskStatus.FAILED
        self.error = error
        self.completed_at = datetime.now().isoformat()
    
    class Config:
        use_enum_values = True


class HumanInteraction(BaseModel):
    """Human-in-the-loop interaction record."""
    interaction_id: str
    type: InteractionType
    prompt: str
    response: Optional[str] = None
    status: InteractionStatus = InteractionStatus.PENDING
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    class Config:
        use_enum_values = True


class Plan(BaseModel):
    """Resolution plan for the query."""
    steps: List[str] = Field(default_factory=list)
    agents_required: List[str] = Field(default_factory=list)
    estimated_complexity: Complexity = Complexity.MEDIUM
    requires_human_input: bool = False
    filters: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    class Config:
        use_enum_values = True


class MarieState(BaseModel):
    """
    Enhanced state for MARIE multi-agent system with Pydantic validation.
    
    Provides:
    - Type safety with validation
    - Default values
    - Serialization/deserialization
    - State transition methods
    """
    
    # Request metadata
    request_id: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M%S"))
    user_query: str
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    # Planning
    plan: Optional[Plan] = None
    current_step: int = 0
    parsed_query: Optional[Dict[str, Any]] = None
    needs_rag: Optional[bool] = None
    
    # Tasks and execution
    tasks: List[Task] = Field(default_factory=list)
    
    # Evidence collection
    evidence_map: Dict[str, List[Evidence]] = Field(default_factory=dict)
    
    # Human interaction
    human_interactions: List[HumanInteraction] = Field(default_factory=list)
    requires_human_review: bool = False
    
    # Results
    entities_resolved: Dict[str, Any] = Field(default_factory=dict)
    retrieved_data: List[Dict[str, Any]] = Field(default_factory=list)
    computed_metrics: Dict[str, Any] = Field(default_factory=dict)
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Final output
    final_answer: Optional[str] = None
    report: Optional[str] = None
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    confidence_assessment: Optional[Dict[str, Any]] = None
    
    # Magentic additions
    quality_report: Optional[Dict[str, Any]] = None
    refinement_count: int = 0
    replan_reason: Optional[str] = None
    
    # Observability
    trace_id: Optional[str] = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M%S%f"))
    execution_time_seconds: Optional[float] = None
    
    def update_timestamp(self) -> None:
        """Update the last modified timestamp."""
        self.updated_at = datetime.now().isoformat()
    
    def add_task(self, agent: str, input_data: Dict[str, Any]) -> Task:
        """
        Add a new task to the state.
        
        Args:
            agent: Agent name
            input_data: Task input
            
        Returns:
            Created task
        """
        task = Task(
            task_id=f"{agent}_{len(self.tasks)}",
            agent=agent,
            input=input_data
        )
        self.tasks.append(task)
        self.update_timestamp()
        return task
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None
    
    def add_evidence(self, claim_id: str, evidence: Evidence) -> None:
        """Add evidence for a claim."""
        if claim_id not in self.evidence_map:
            self.evidence_map[claim_id] = []
        self.evidence_map[claim_id].append(evidence)
        self.update_timestamp()
    
    def add_human_interaction(self, interaction: HumanInteraction) -> None:
        """Add human interaction."""
        self.human_interactions.append(interaction)
        self.update_timestamp()
    
    def increment_refinement(self, reason: str) -> None:
        """Increment refinement count and record reason."""
        self.refinement_count += 1
        self.replan_reason = reason
        self.update_timestamp()
    
    def set_confidence(self, score: float) -> None:
        """Set overall confidence score."""
        if not 0.0 <= score <= 1.0:
            raise ValueError("Confidence score must be between 0.0 and 1.0")
        self.confidence_score = score
        self.update_timestamp()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (LangGraph compatible)."""
        return self.dict()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarieState":
        """Create from dictionary (LangGraph compatible)."""
        return cls(**data)
    
    class Config:
        use_enum_values = True
        validate_assignment = True  # Validate on attribute assignment


# Backwards compatibility - convert to dict for LangGraph
def create_state(query: str, **kwargs) -> Dict[str, Any]:
    """
    Create initial state (backwards compatible).
    
    Args:
        query: User query
        **kwargs: Additional state fields
        
    Returns:
        State dict
    """
    state = MarieState(user_query=query, **kwargs)
    return state.to_dict()
