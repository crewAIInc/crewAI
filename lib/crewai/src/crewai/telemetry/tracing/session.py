"""Execution-scoped providers and lifecycle for event-created spans."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
import logging
import threading
from typing import Any, cast

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExportResult,
    SpanExporter,
)

from crewai.events.event_bus import CrewAIEventsBus, crewai_event_bus
from crewai.events.event_context import _event_id_stack
from crewai.execution import clear_execution_uuid, set_execution_uuid
from crewai.telemetry.tracing.context import (
    TelemetryExecutionContext,
    _telemetry_context,
    _trace_session,
)
from crewai.version import get_crewai_version


logger = logging.getLogger(__name__)
MAX_EXPORT_BATCH_SIZE = 200


class _CompletedSpanReference(trace.NonRecordingSpan):
    """Retain parent identity and policy timing without retaining payloads."""

    def __init__(self, span: ReadableSpan):
        super().__init__(span.get_span_context() or trace.INVALID_SPAN_CONTEXT)
        self.start_time, self.end_time = span.start_time, span.end_time


class ExecutionAttributes(SpanProcessor):
    """Stamp execution identity and release completed span payloads."""

    def __init__(self, execution_uuid: str, attributes: dict[str, Any] | None = None):
        if not execution_uuid:
            raise ValueError("execution_uuid must be non-empty")
        self.attributes = {
            **(attributes or {}),
            "crewai.execution_uuid": execution_uuid,
        }
        self.context: TelemetryExecutionContext | None = None

    def on_start(self, span: trace.Span, parent_context: Context | None = None) -> None:
        span.set_attributes(self.attributes)

    def on_end(self, span: ReadableSpan) -> None:
        if self.context is None:
            return
        reference = _CompletedSpanReference(span)
        event_id = (span.attributes or {}).get("event_id")
        with self.context._span_lock:
            if isinstance(event_id, str) and event_id in self.context._span_refs:
                self.context._span_refs[event_id] = reference
            root = self.context.root_span
            if root is not None and root.get_span_context() == span.get_span_context():
                self.context.root_span = reference

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


class _ResourceSpan:
    def __init__(self, span: ReadableSpan, resource: Resource):
        self._span, self.resource = span, resource

    def __getattr__(self, name: str) -> Any:
        return getattr(self._span, name)


class ResourceSpanExporter(SpanExporter):
    """Apply a collector-specific resource while keeping the original spans."""

    def __init__(self, delegate: SpanExporter, resource: Resource):
        self._delegate, self._resource = delegate, resource

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return self._delegate.export(
            [cast(ReadableSpan, _ResourceSpan(span, self._resource)) for span in spans]
        )

    def shutdown(self) -> None:
        self._delegate.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._delegate.force_flush(timeout_millis)


def otlp_exporter(
    endpoint: str,
    headers: dict[str, str] | None = None,
    certificate_file: str | None = None,
) -> OTLPSpanExporter:
    """Build the shared OTLP/HTTP transport without changing global OTel."""
    return OTLPSpanExporter(
        endpoint=endpoint,
        headers=headers or {},
        certificate_file=certificate_file,
        timeout=5,
    )


class TraceSession:
    """Own an event-driven trace and only its own event-bus subscriptions.

    Exporters and redaction processors are supplied by the standalone or hosted
    caller. Existing application listeners and global OTel providers stay active.
    """

    def __init__(
        self,
        execution_uuid: str,
        exporters: Sequence[SpanExporter] = (),
        *,
        resource: Resource | None = None,
        attributes: dict[str, Any] | None = None,
        processors: Sequence[SpanProcessor] = (),
        automation_name: str = "crewai",
        log_emitter: Callable[..., None] | None = None,
        event_bus: CrewAIEventsBus | None = None,
        providers: Any = None,
    ) -> None:
        self.event_bus = event_bus or crewai_event_bus
        self._providers = providers
        self.tracer_provider = (
            providers.tracer_provider
            if providers is not None
            else TracerProvider(
                resource=resource or Resource({"service.name": "crewai"})
            )
        )
        self.execution_attributes = ExecutionAttributes(execution_uuid, attributes)
        self.tracer_provider.add_span_processor(self.execution_attributes)
        for processor in processors:
            self.tracer_provider.add_span_processor(processor)
        for exporter in exporters:
            self.add_exporter(exporter)
        self.context = TelemetryExecutionContext(
            kickoff_id=execution_uuid,
            automation_name=automation_name,
            tracer=self.get_tracer(),
        )
        self.execution_attributes.context = self.context
        self.log_emitter = log_emitter or (
            providers.emit_log if providers is not None else None
        )
        self._closed = False
        self._accept_events = True
        self._running_handlers = 0
        self._handlers_drained = threading.Condition()
        from crewai.telemetry.tracing.handlers import register_handlers

        self._registrations = register_handlers(self, self.context, self.event_bus)

    def add_exporter(self, exporter: SpanExporter) -> None:
        """Attach a destination using Wharf's 200-span request cap."""
        self.tracer_provider.add_span_processor(
            BatchSpanProcessor(exporter, max_export_batch_size=MAX_EXPORT_BATCH_SIZE)
        )

    def get_tracer(self, name: str | None = None) -> trace.Tracer:
        if self._providers is not None:
            return cast(
                trace.Tracer,
                self._providers.get_tracer()
                if name is None
                else self._providers.get_tracer(name),
            )
        return self.tracer_provider.get_tracer(name or "crewai", get_crewai_version())

    def emit_log(self, *args: Any, **kwargs: Any) -> None:
        if self.log_emitter is not None:
            self.log_emitter(*args, **kwargs)

    def _run_handler(
        self, handler: Callable[..., None], source: Any, event: Any
    ) -> None:
        with self._handlers_drained:
            if not self._accept_events:
                return
            self._running_handlers += 1
        try:
            handler(source, event)
        finally:
            with self._handlers_drained:
                self._running_handlers -= 1
                if self._running_handlers == 0:
                    if self._closed:
                        self._clear_context()
                    self._handlers_drained.notify_all()

    @contextmanager
    def activate(self) -> Iterator[TelemetryExecutionContext]:
        """Bind this session; the caller retains ownership of its lifetime."""
        if self._closed:
            raise RuntimeError("Cannot activate a closed trace session")
        session_token = _trace_session.set(self)
        context_token = _telemetry_context.set(self.context)
        execution_token = set_execution_uuid(self.context.kickoff_id)
        event_scope_token = _event_id_stack.set(_event_id_stack.get())
        try:
            yield self.context
        finally:
            _event_id_stack.reset(event_scope_token)
            clear_execution_uuid(execution_token)
            _telemetry_context.reset(context_token)
            _trace_session.reset(session_token)

    def flush(self, timeout_millis: int = 30000) -> bool:
        if self._providers is not None:
            return bool(self._providers.flush(timeout_millis))
        return self.tracer_provider.force_flush(timeout_millis)

    def finish_spans(self) -> bool:
        """Drain event handlers, then close spans missing completion events."""
        drained = self.event_bus.flush(timeout=30.0)
        if not drained:
            logger.warning("Execution trace handlers did not finish before timeout")
        with self._handlers_drained:
            self._accept_events = False
            # A stalled unrelated listener must not keep the trace session alive.
            # Already-running trace callbacks get one final bounded drain window.
            if not self._handlers_drained.wait_for(
                lambda: self._running_handlers == 0, timeout=30.0
            ):
                return False
        for span in list(self.context.active_spans.values()):
            if span.is_recording():
                span.set_status(
                    trace.Status(
                        trace.StatusCode.ERROR,
                        "Span orphaned — execution ended before completion event",
                    )
                )
                span.end()
        self.context.active_spans.clear()
        return drained

    def shutdown(self, timeout_millis: int = 30000) -> bool:
        """Drain execution events, export, and remove only this session's handlers."""
        if self._closed:
            return True
        drained = False
        try:
            drained = self.finish_spans()
            return self.flush(timeout_millis) and drained
        finally:
            self._closed = True
            self._accept_events = False
            for event_type, handler in self._registrations:
                self.event_bus.off(event_type, handler)
            self._registrations.clear()
            try:
                if self._providers is not None:
                    self._providers.shutdown(timeout_millis)
                else:
                    self.tracer_provider.shutdown()
            finally:
                with self._handlers_drained:
                    if self._running_handlers == 0:
                        self._clear_context()

    def _clear_context(self) -> None:
        # If a callback exceeded the bounded drain, its finally block performs
        # this cleanup when it exits instead of racing its span mutations.
        self.context.active_spans.clear()
        self.context._span_refs.clear()
        self.context._span_ready.clear()
        self.context.pending_span_ends.clear()
        self.context.agent_llm_call_counts.clear()
        self.context._agent_llm_ready.clear()
        self.context.flow_crew_usage_metrics.clear()
        self.context.root_span = None


@contextmanager
def telemetry_session(
    kickoff_id: str,
    automation_name: str,
    exporters: Sequence[SpanExporter] = (),
    event_bus: CrewAIEventsBus | None = None,
    providers: Any = None,
    parent_otel_context: tuple[int, int] | None = None,
    execution_id: str | None = None,
    resume_feedback: str | None = None,
    principal: dict[str, Any] | None = None,
    origin: str | None = None,
    *,
    resource: Resource | None = None,
    attributes: dict[str, Any] | None = None,
    processors: Sequence[SpanProcessor] = (),
    log_emitter: Callable[..., None] | None = None,
    pii_redactor: Any = None,
) -> Iterator[TelemetryExecutionContext]:
    """Capture a crew or flow using the enterprise event/session lifecycle.

    Hosts supply verified identity, collectors, resources, and optional redaction
    processors. ``pii_redactor`` is exposed to host log hooks via the context;
    redacting span attributes is the supplied processors' responsibility.
    """
    attrs = dict(attributes or {})
    if principal:
        if principal.get("type") is not None:
            attrs["crewai.principal.type"] = principal["type"]
        if principal.get("id") is not None:
            attrs["crewai.principal.id"] = str(principal["id"])
            if principal.get("type") == "user":
                attrs["enduser.id"] = str(principal["id"])
    if origin:
        attrs["crewai.execution.origin"] = origin
    session = (
        providers
        if isinstance(providers, TraceSession)
        else TraceSession(
            kickoff_id,
            exporters,
            resource=resource,
            attributes=attrs,
            processors=processors,
            automation_name=automation_name,
            log_emitter=log_emitter,
            event_bus=event_bus,
            providers=providers,
        )
    )
    session.execution_attributes.attributes.update(attrs)
    ctx = session.context
    ctx.execution_id = execution_id
    ctx.principal = principal
    ctx.origin = origin
    ctx.parent_otel_context = parent_otel_context
    ctx.otel_resume_context = parent_otel_context
    ctx.resume_feedback = resume_feedback
    ctx.pii_redactor = pii_redactor
    try:
        with session.activate():
            yield ctx
    finally:
        session.shutdown()
