"""
Observability system for MARIE multi-agent system.

Provides distributed tracing, metrics collection, and structured logging.
Inspired by OpenTelemetry patterns and LangChain best practices.
"""

import time
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# Context variable for trace propagation
_trace_context: ContextVar[Optional["TraceContext"]] = ContextVar("trace_context", default=None)


class SpanKind(str, Enum):
    """Span kind following OpenTelemetry conventions."""
    AGENT = "agent"
    LLM = "llm"
    RETRIEVAL = "retrieval"
    DATABASE = "database"
    CACHE = "cache"
    INTERNAL = "internal"


class SpanStatus(str, Enum):
    """Span status."""
    OK = "ok"
    ERROR = "error"
    UNSET = "unset"


@dataclass
class Span:
    """
    A span represents a single operation with timing and metadata.
    
    Following OpenTelemetry semantic conventions.
    """
    span_id: str
    name: str
    kind: SpanKind
    start_time: float
    parent_span_id: Optional[str] = None
    end_time: Optional[float] = None
    status: SpanStatus = SpanStatus.UNSET
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def duration_ms(self) -> Optional[float]:
        """Get duration in milliseconds."""
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None
    
    def set_attribute(self, key: str, value: Any) -> None:
        """Set span attribute."""
        self.attributes[key] = value
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add event to span."""
        event = {
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {}
        }
        self.events.append(event)
    
    def set_status(self, status: SpanStatus, description: Optional[str] = None) -> None:
        """Set span status."""
        self.status = status
        if description:
            self.set_attribute("status.description", description)
    
    def end(self) -> None:
        """End the span."""
        if not self.end_time:
            self.end_time = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "span_id": self.span_id,
            "name": self.name,
            "kind": self.kind.value,
            "parent_span_id": self.parent_span_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "attributes": self.attributes,
            "events": self.events
        }


@dataclass
class TraceContext:
    """
    Trace context for distributed tracing.
    
    Maintains trace_id and current span stack for proper parent-child relationships.
    """
    trace_id: str
    request_id: str
    user_query: str
    spans: List[Span] = field(default_factory=list)
    current_span: Optional[Span] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None
    ) -> Span:
        """Start a new span."""
        span = Span(
            span_id=str(uuid.uuid4())[:8],
            name=name,
            kind=kind,
            start_time=time.time(),
            parent_span_id=self.current_span.span_id if self.current_span else None,
            attributes=attributes or {}
        )
        
        # Add trace context
        span.set_attribute("trace_id", self.trace_id)
        span.set_attribute("request_id", self.request_id)
        
        self.spans.append(span)
        self.current_span = span
        
        logger.debug(f"Started span: {name} ({span.span_id})")
        return span
    
    def end_span(self) -> None:
        """End current span and pop to parent."""
        if self.current_span:
            self.current_span.end()
            
            # Log span completion
            duration = self.current_span.duration_ms
            logger.debug(
                f"Ended span: {self.current_span.name} "
                f"({self.current_span.span_id}) - {duration:.2f}ms"
            )
            
            # Find parent span
            parent_id = self.current_span.parent_span_id
            if parent_id:
                for span in reversed(self.spans):
                    if span.span_id == parent_id and not span.end_time:
                        self.current_span = span
                        return
            
            self.current_span = None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics from all spans."""
        metrics = {
            "trace_id": self.trace_id,
            "request_id": self.request_id,
            "total_spans": len(self.spans),
            "total_duration_ms": 0.0,
            "spans_by_kind": {},
            "spans_by_status": {},
            "agent_timings": {},
            "error_count": 0
        }
        
        for span in self.spans:
            if span.duration_ms:
                metrics["total_duration_ms"] += span.duration_ms
                
                # By kind
                kind = span.kind.value
                if kind not in metrics["spans_by_kind"]:
                    metrics["spans_by_kind"][kind] = 0
                metrics["spans_by_kind"][kind] += 1
                
                # By status
                status = span.status.value
                if status not in metrics["spans_by_status"]:
                    metrics["spans_by_status"][status] = 0
                metrics["spans_by_status"][status] += 1
                
                if status == SpanStatus.ERROR.value:
                    metrics["error_count"] += 1
                
                # Agent timings
                if span.kind == SpanKind.AGENT:
                    metrics["agent_timings"][span.name] = span.duration_ms
        
        return metrics
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trace_id": self.trace_id,
            "request_id": self.request_id,
            "user_query": self.user_query,
            "spans": [s.to_dict() for s in self.spans],
            "metadata": self.metadata,
            "metrics": self.get_metrics()
        }


def create_trace_context(request_id: str, user_query: str) -> TraceContext:
    """
    Create and activate a new trace context.
    
    Args:
        request_id: Request identifier
        user_query: User query
        
    Returns:
        TraceContext instance
    """
    trace_id = str(uuid.uuid4())
    context = TraceContext(
        trace_id=trace_id,
        request_id=request_id,
        user_query=user_query
    )
    
    _trace_context.set(context)
    logger.info(f"Created trace context: {trace_id} for request {request_id}")
    
    return context


def get_trace_context() -> Optional[TraceContext]:
    """Get current trace context."""
    return _trace_context.get()


def start_span(
    name: str,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Optional[Dict[str, Any]] = None
) -> Optional[Span]:
    """
    Start a span in current trace context.
    
    Args:
        name: Span name
        kind: Span kind
        attributes: Optional attributes
        
    Returns:
        Span if trace context exists, None otherwise
    """
    context = get_trace_context()
    if context:
        return context.start_span(name, kind, attributes)
    return None


def end_span() -> None:
    """End current span in trace context."""
    context = get_trace_context()
    if context:
        context.end_span()


def add_span_event(name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
    """Add event to current span."""
    context = get_trace_context()
    if context and context.current_span:
        context.current_span.add_event(name, attributes)


def set_span_attribute(key: str, value: Any) -> None:
    """Set attribute on current span."""
    context = get_trace_context()
    if context and context.current_span:
        context.current_span.set_attribute(key, value)


def set_span_status(status: SpanStatus, description: Optional[str] = None) -> None:
    """Set status on current span."""
    context = get_trace_context()
    if context and context.current_span:
        context.current_span.set_status(status, description)


class traced:
    """
    Decorator for tracing function execution.
    
    Usage:
        @traced(name="my_function", kind=SpanKind.AGENT)
        def my_function():
            ...
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        kind: SpanKind = SpanKind.INTERNAL,
        capture_args: bool = False
    ):
        self.name = name
        self.kind = kind
        self.capture_args = capture_args
    
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            span_name = self.name or func.__name__
            
            # Start span
            attributes = {}
            if self.capture_args:
                attributes["args"] = str(args)[:100]  # Truncate
                attributes["kwargs"] = str(kwargs)[:100]
            
            span = start_span(span_name, self.kind, attributes)
            
            try:
                result = func(*args, **kwargs)
                
                if span:
                    set_span_status(SpanStatus.OK)
                
                return result
                
            except Exception as e:
                if span:
                    set_span_status(SpanStatus.ERROR, str(e))
                    add_span_event("exception", {
                        "exception.type": type(e).__name__,
                        "exception.message": str(e)
                    })
                raise
            finally:
                end_span()
        
        return wrapper


def get_trace_metrics() -> Optional[Dict[str, Any]]:
    """Get metrics from current trace."""
    context = get_trace_context()
    if context:
        return context.get_metrics()
    return None


def export_trace() -> Optional[Dict[str, Any]]:
    """Export current trace context."""
    context = get_trace_context()
    if context:
        return context.to_dict()
    return None
