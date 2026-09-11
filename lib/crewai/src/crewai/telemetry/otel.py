"""Native OpenTelemetry instrumentation surface for crewAI.

This module exposes a thin wrapper over the OpenTelemetry API.
crewAI emits spans through :func:`operation` for kickoffs, tasks, agents,
tools, LLM calls, memory, knowledge, MCP, and A2A delegation.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import (
    Link,
    Span,
    SpanContext,
    SpanKind,
    Status,
    StatusCode,
    TraceFlags,
)

from crewai.events.event_context import get_current_parent_id
from crewai.execution import get_execution_uuid
from crewai.telemetry.tracing.context import get_trace_session


_TRACER_NAME = "crewai"


def _tracer() -> trace.Tracer:
    """Prefer the execution's tracer; otherwise preserve application OTel wiring.

    Always re-resolves so user code that installs a TracerProvider after
    crewAI is imported still gets recording spans.
    """
    if session := get_trace_session():
        return session.get_tracer(_TRACER_NAME)
    return trace.get_tracer(_TRACER_NAME)


@contextmanager
def operation(
    name: str,
    attributes: dict[str, Any] | None = None,
    *,
    links: list[Link] | None = None,
    expected_exceptions: tuple[type[BaseException], ...] = (),
    enabled: bool = True,
) -> Iterator[Span]:
    """Open a span around an operation.

    Any :class:`Exception` escaping the block is recorded as an
    ``exception`` event and the span status is set to ``ERROR``.
    ``BaseException`` subclasses outside :class:`Exception`
    (:class:`KeyboardInterrupt`, :class:`SystemExit`,
    :class:`asyncio.CancelledError`, :class:`GeneratorExit`) pass through
    unrecorded — they're control flow, not failures.

    Args:
        name: Span name (e.g. ``"execute crew"``).  Follow the
            ``"<verb> <subject>"`` convention used elsewhere in this module.
        attributes: Optional dict of attributes to set on span start.
            Keys should follow the ``crewai.<component>.<field>`` pattern.
        links: Optional list of :class:`Link` references.  Used for
            HITL resume to relate the resumed trace back to the paused one
            via :func:`follows_from`.
        expected_exceptions: Exception types that represent expected
            control flow rather than failures (e.g.
            :class:`HumanFeedbackPending`).  When the block raises one of
            these the span's status stays ``UNSET``, no ``exception``
            event is recorded, and the exception re-raises so the caller
            sees normal control flow.

    Yields:
        The active :class:`Span`.  Callers may attach additional
        attributes or events to it as the operation progresses.
    """
    if not enabled:
        yield trace.get_current_span()
        return
    attrs: dict[str, Any] = dict(attributes or {})
    execution_uuid = get_execution_uuid()
    if execution_uuid and "crewai.execution_uuid" not in attrs:
        attrs["crewai.execution_uuid"] = execution_uuid

    session = get_trace_session()
    if session:
        name = {
            "execute flow method": "call method",
            "resume flow": "execute flow",
            "remember memory": "save memory",
            "recall memory": "query memory",
            "agent reason": "agent reasoning",
            "guard llm": "evaluate guardrail",
        }.get(name, name)
    span = None
    if session:
        # Some lifecycle events precede the operation wrapper. Adopt their
        # span rather than creating a second row for the same operation.
        candidate = session.context.active_spans.get(get_current_parent_id() or "")
        if (
            candidate is not None
            and getattr(candidate, "name", None) == name
            and candidate.is_recording()
            and candidate.get_span_context().span_id
            not in session.context.operation_spans
        ):
            span = candidate
            span.set_attributes(attrs)
            for link in links or []:
                span.add_link(link.context, link.attributes)
    if span is None:
        kind = (
            SpanKind.CLIENT
            if session and name in {"call llm", "execute lite agent"}
            else SpanKind.INTERNAL
        )
        span = _tracer().start_span(
            name, attributes=attrs, links=links or [], kind=kind
        )
    if session:
        session.context.operation_spans.add(span.get_span_context().span_id)
        if session.context.root_span is None:
            session.context.root_span = span

    with trace.use_span(
        span,
        end_on_exit=session is None,
        record_exception=False,
        set_status_on_exception=False,
    ):
        try:
            yield span
        except expected_exceptions:
            raise
        except Exception as exc:
            span.record_exception(exc, escaped=True)
            span.set_status(Status(StatusCode.ERROR, f"{type(exc).__name__}: {exc}"))
            raise
        else:
            if getattr(
                getattr(span, "status", None), "status_code", None
            ) != StatusCode.ERROR and (
                session is None or span not in session.context._span_refs.values()
            ):
                span.set_status(Status(StatusCode.OK))
        finally:
            if session:
                span_id = span.get_span_context().span_id
                session.context.operation_spans.discard(span_id)
                end_time = session.context.completed_operations.pop(span_id, None)
                if end_time is not None:
                    span.end(end_time=end_time)
                elif span not in session.context._span_refs.values():
                    span.end()


def follows_from(
    trace_id: int,
    span_id: int,
    *,
    is_remote: bool = False,
    trace_flags: TraceFlags | None = None,
) -> Link:
    """Build a FOLLOWS_FROM-style :class:`Link` for HITL resume continuity.

    Args:
        trace_id: Trace ID of the paused operation's span.
        span_id: Span ID of the paused operation's span.
        is_remote: Whether the linked span context came from outside this
            process.  Default ``False`` matches crewAI OSS's in-process
            resume flow (same Python process pauses and resumes).  Cross-
            process resumers (e.g. an enterprise Celery worker that picks
            up a flow paused by a different worker) should pass ``True``
            so backends render the edge as crossing a process boundary
            and so samplers treat the parent context as an inbound
            carrier rather than a local span.
        trace_flags: Optional :class:`TraceFlags` for the linked span.
            Default ``None`` resolves to ``TraceFlags.SAMPLED`` so backends
            render the link reliably even when the original sampling
            decision was not persisted.  Callers that persist the
            original flags at pause time should pass them here.

    Returns:
        A :class:`Link` carrying a :class:`SpanContext` for the paused
        span, suitable to pass via the ``links=`` kwarg of
        :func:`operation`.
    """
    span_ctx = SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        is_remote=is_remote,
        trace_flags=trace_flags
        if trace_flags is not None
        else TraceFlags(TraceFlags.SAMPLED),
    )
    return Link(span_ctx, attributes={"crewai.link.type": "follows_from"})


__all__ = ["follows_from", "operation"]
