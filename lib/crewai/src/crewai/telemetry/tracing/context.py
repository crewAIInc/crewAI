"""Execution-local trace state, independent of any exporter or hosting platform."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
import threading
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from opentelemetry.trace import Span, Tracer

    from crewai.telemetry.tracing.session import TraceSession


@dataclass
class PendingSpanEnd:
    attributes: dict[str, Any]
    error: str | BaseException | None
    end_time_ns: int
    duration_attr: str | None = None
    end_event: Any = None


@dataclass
class TelemetryExecutionContext:
    kickoff_id: str
    automation_name: str
    tracer: Tracer
    execution_id: str | None = None
    principal: dict[str, Any] | None = None
    origin: str | None = None
    active_spans: dict[str, Span] = field(default_factory=dict)
    _span_refs: dict[str, Span] = field(default_factory=dict)
    pending_span_ends: dict[str, PendingSpanEnd] = field(default_factory=dict)
    _span_lock: threading.Lock = field(default_factory=threading.Lock)
    root_span: Span | None = None
    agent_llm_call_counts: dict[str, int] = field(default_factory=dict)
    flow_crew_usage_metrics: dict[str, dict[str, int]] = field(default_factory=dict)
    otel_resume_context: tuple[int, int] | None = None
    parent_otel_context: tuple[int, int] | None = None
    resume_feedback: str | None = None
    pii_redactor: Any = None
    operation_spans: set[int] = field(default_factory=set)
    completed_operations: dict[int, int] = field(default_factory=dict)


_telemetry_context: ContextVar[TelemetryExecutionContext | None] = ContextVar(
    "crewai_trace_context", default=None
)
_trace_session: ContextVar[TraceSession | None] = ContextVar(
    "crewai_trace_session", default=None
)


def get_telemetry_context() -> TelemetryExecutionContext | None:
    """Return the current execution's span and metadata state."""
    return _telemetry_context.get()


def get_trace_session() -> TraceSession | None:
    """Return the current execution's tracing session, if configured."""
    return _trace_session.get()


def get_execution_principal() -> dict[str, Any] | None:
    """Return the host-verified principal for this execution."""
    ctx = get_telemetry_context()
    return ctx.principal if ctx else None
