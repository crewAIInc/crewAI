"""Execution-scoped state for event-driven tracing."""

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
    """Stores end-span data when the completion event arrives before the span is created."""

    attributes: dict[str, Any]
    error: str | BaseException | None
    end_time_ns: int
    duration_attr: str | None = None
    end_event: Any = None


@dataclass
class TelemetryExecutionContext:
    """Execution-scoped context for telemetry data.

    Event-bus handlers run on a 10-worker thread pool, so a child-event
    handler can execute before the parent-event handler stores its span.
    ``_span_lock`` and ``_span_ready`` provide deterministic
    synchronization: ``_store_span`` signals the event, and
    ``_get_parent_context`` / ``_end_span`` wait on it.
    """

    kickoff_id: str
    automation_name: str
    tracer: Tracer
    execution_id: str | None = None
    # Who executed the automation, resolved by AMP. None when the caller's AMP
    # predates principal resolution - the execution is then simply unattributed.
    principal: dict[str, Any] | None = None
    # Where the run came from (ui/schedule/api/hitl-resume/replay/trigger),
    # from AMP's X-Crewai-Execution-Origin header. None when not supplied.
    origin: str | None = None
    active_spans: dict[str, Span] = field(default_factory=dict)
    _span_refs: dict[str, Span] = field(default_factory=dict)
    pending_span_ends: dict[str, PendingSpanEnd] = field(default_factory=dict)
    _span_lock: threading.Lock = field(default_factory=threading.Lock)
    _span_ready: dict[str, threading.Event] = field(default_factory=dict)
    root_span: Span | None = None
    agent_llm_call_counts: dict[str, int] = field(default_factory=dict)
    _agent_llm_ready: dict[str, threading.Event] = field(default_factory=dict)
    flow_crew_usage_metrics: dict[str, dict[str, int]] = field(default_factory=dict)
    otel_resume_context: tuple[int, int] | None = None
    parent_otel_context: tuple[int, int] | None = None
    resume_feedback: str | None = None

    def _get_or_create_event(self, event_id: str) -> threading.Event:
        """Get or create a threading.Event for the given event_id (must hold _span_lock)."""
        ev = self._span_ready.get(event_id)
        if ev is None:
            ev = threading.Event()
            self._span_ready[event_id] = ev
        return ev


_telemetry_context: ContextVar[TelemetryExecutionContext | None] = ContextVar(
    "_telemetry_context", default=None
)


def get_telemetry_context() -> TelemetryExecutionContext | None:
    """Get the current telemetry execution context."""
    return _telemetry_context.get()


def get_execution_principal() -> dict[str, Any] | None:
    """Return the principal (who executed) for the active telemetry session.

    None outside a session or when the run is unattributed - callers omit the
    ``executed_by`` field in that case so older AMP versions are unaffected.
    """
    ctx = get_telemetry_context()
    return ctx.principal if ctx else None


_trace_session: ContextVar[TraceSession | None] = ContextVar(
    "crewai_trace_session", default=None
)


def get_trace_session() -> TraceSession | None:
    """Return the tracing session owned by the active execution."""
    return _trace_session.get()
