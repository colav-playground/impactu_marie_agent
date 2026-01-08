"""
Metrics collection system for MARIE agents.

Tracks performance, usage, and quality metrics.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
from threading import Lock
import logging

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """
    Thread-safe metrics collector.
    
    Tracks:
    - Counters (monotonic increasing)
    - Gauges (point-in-time values)
    - Histograms (distribution of values)
    - Timers (duration measurements)
    """
    
    def __init__(self):
        self._lock = Lock()
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._timers: Dict[str, List[MetricPoint]] = defaultdict(list)
        
        logger.info("Metrics collector initialized")
    
    def increment_counter(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Increment a counter metric."""
        with self._lock:
            key = self._make_key(name, labels)
            self._counters[key] += value
    
    def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Set a gauge metric."""
        with self._lock:
            key = self._make_key(name, labels)
            self._gauges[key] = value
    
    def record_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a value in histogram."""
        with self._lock:
            key = self._make_key(name, labels)
            self._histograms[key].append(value)
    
    def record_timer(
        self,
        name: str,
        duration_ms: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a timing measurement."""
        with self._lock:
            key = self._make_key(name, labels)
            point = MetricPoint(
                timestamp=datetime.now().timestamp(),
                value=duration_ms,
                labels=labels or {}
            )
            self._timers[key].append(point)
    
    def _make_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Create metric key with labels."""
        if not labels:
            return name
        
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def get_counter(self, name: str) -> float:
        """Get counter value."""
        with self._lock:
            return self._counters.get(name, 0.0)
    
    def get_gauge(self, name: str) -> Optional[float]:
        """Get gauge value."""
        with self._lock:
            return self._gauges.get(name)
    
    def get_histogram_stats(self, name: str) -> Dict[str, float]:
        """Get histogram statistics."""
        with self._lock:
            values = self._histograms.get(name, [])
            
            if not values:
                return {}
            
            sorted_values = sorted(values)
            count = len(sorted_values)
            
            return {
                "count": count,
                "min": sorted_values[0],
                "max": sorted_values[-1],
                "mean": sum(sorted_values) / count,
                "median": sorted_values[count // 2],
                "p95": sorted_values[int(count * 0.95)] if count > 1 else sorted_values[0],
                "p99": sorted_values[int(count * 0.99)] if count > 1 else sorted_values[0]
            }
    
    def get_timer_stats(self, name: str) -> Dict[str, Any]:
        """Get timer statistics."""
        with self._lock:
            points = self._timers.get(name, [])
            
            if not points:
                return {}
            
            values = [p.value for p in points]
            sorted_values = sorted(values)
            count = len(sorted_values)
            
            return {
                "count": count,
                "min_ms": sorted_values[0],
                "max_ms": sorted_values[-1],
                "mean_ms": sum(sorted_values) / count,
                "median_ms": sorted_values[count // 2],
                "p95_ms": sorted_values[int(count * 0.95)] if count > 1 else sorted_values[0],
                "p99_ms": sorted_values[int(count * 0.99)] if count > 1 else sorted_values[0],
                "total_ms": sum(sorted_values)
            }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {
                    name: self.get_histogram_stats(name)
                    for name in self._histograms.keys()
                },
                "timers": {
                    name: self.get_timer_stats(name)
                    for name in self._timers.keys()
                }
            }
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._timers.clear()
            logger.info("All metrics reset")


# Global metrics collector
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


# Convenience functions
def increment(name: str, value: float = 1.0, **labels) -> None:
    """Increment counter metric."""
    get_metrics_collector().increment_counter(name, value, labels or None)


def gauge(name: str, value: float, **labels) -> None:
    """Set gauge metric."""
    get_metrics_collector().set_gauge(name, value, labels or None)


def histogram(name: str, value: float, **labels) -> None:
    """Record histogram value."""
    get_metrics_collector().record_histogram(name, value, labels or None)


def timer(name: str, duration_ms: float, **labels) -> None:
    """Record timing."""
    get_metrics_collector().record_timer(name, duration_ms, labels or None)


def get_metrics() -> Dict[str, Any]:
    """Get all metrics."""
    return get_metrics_collector().get_all_metrics()


# Agent-specific metrics
def record_agent_execution(
    agent_name: str,
    duration_ms: float,
    success: bool,
    error: Optional[str] = None
) -> None:
    """Record agent execution metrics."""
    labels = {"agent": agent_name, "status": "success" if success else "error"}
    
    increment("agent.executions.total", **labels)
    timer("agent.execution.duration", duration_ms, **labels)
    
    if not success and error:
        increment("agent.errors.total", agent=agent_name, error_type=type(error).__name__)


def record_llm_call(
    provider: str,
    model: str,
    duration_ms: float,
    input_tokens: int,
    output_tokens: int,
    success: bool
) -> None:
    """Record LLM call metrics."""
    labels = {"provider": provider, "model": model}
    
    increment("llm.calls.total", **labels)
    timer("llm.call.duration", duration_ms, **labels)
    histogram("llm.tokens.input", input_tokens, **labels)
    histogram("llm.tokens.output", output_tokens, **labels)
    
    if success:
        increment("llm.calls.success", **labels)
    else:
        increment("llm.calls.error", **labels)


def record_cache_operation(
    cache_name: str,
    operation: str,
    hit: bool,
    duration_ms: float
) -> None:
    """Record cache operation metrics."""
    labels = {"cache": cache_name, "operation": operation}
    
    increment("cache.operations.total", **labels)
    
    if hit:
        increment("cache.hits.total", cache=cache_name)
    else:
        increment("cache.misses.total", cache=cache_name)
    
    timer("cache.operation.duration", duration_ms, **labels)


def record_opensearch_query(
    index: str,
    duration_ms: float,
    result_count: int,
    success: bool
) -> None:
    """Record OpenSearch query metrics."""
    labels = {"index": index}
    
    increment("opensearch.queries.total", **labels)
    timer("opensearch.query.duration", duration_ms, **labels)
    histogram("opensearch.results.count", result_count, **labels)
    
    if success:
        increment("opensearch.queries.success", **labels)
    else:
        increment("opensearch.queries.error", **labels)
