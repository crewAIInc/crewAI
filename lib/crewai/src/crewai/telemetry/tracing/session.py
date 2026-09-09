"""One trace provider and export lifecycle per CrewAI execution."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
import logging
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
    """Keep parent identity and policy timing, but never completed payloads."""

    def __init__(self, span: ReadableSpan):
        super().__init__(span.get_span_context() or trace.INVALID_SPAN_CONTEXT)
        self.start_time, self.end_time = span.start_time, span.end_time


class ExecutionAttributes(SpanProcessor):
    """Stamp verified execution identity before any export processors run."""

    def __init__(self, execution_uuid: str, attributes: dict[str, Any] | None = None):
        self.context: TelemetryExecutionContext | None = None
        if not execution_uuid:
            raise ValueError("execution_uuid must be non-empty")
        self.attributes = {
            **(attributes or {}),
            "crewai.execution_uuid": execution_uuid,
        }

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
    """Apply a collector-specific resource without changing the execution's spans."""

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
    """Build the OTLP/HTTP transport used by all CrewAI execution traces."""
    return OTLPSpanExporter(
        endpoint=endpoint,
        headers=headers or {},
        certificate_file=certificate_file,
        timeout=5,
    )


class TraceSession:
    """Own an isolated execution provider without changing process-global OTel.

    Hosts may supply exporters and processors, including custom collectors and
    redaction. Processors run before batching. Nested kickoffs reuse the active
    session instead of requesting a second grant or closing the host's provider.
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
    ) -> None:
        self.tracer_provider = TracerProvider(
            resource=resource or Resource({"service.name": "crewai"})
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
        self.log_emitter = log_emitter
        self._closed = False
        from crewai.telemetry.tracing.handlers import EVENT_HANDLERS

        self._handlers = EVENT_HANDLERS

    def add_exporter(self, exporter: SpanExporter) -> None:
        """Attach a destination using Wharf's 200-span request cap."""
        self.tracer_provider.add_span_processor(
            BatchSpanProcessor(exporter, max_export_batch_size=MAX_EXPORT_BATCH_SIZE)
        )

    def get_tracer(self, name: str = "crewai") -> trace.Tracer:
        return self.tracer_provider.get_tracer(name, get_crewai_version())

    def emit_log(self, *args: Any, **kwargs: Any) -> None:
        if self.log_emitter is not None:
            self.log_emitter(*args, **kwargs)

    def record_event(self, source: Any, event: Any) -> None:
        if self._closed:
            return
        if handler := self._handlers.get(type(event)):
            try:
                handler(self, self.context, source, event)
            except Exception:
                # Event payloads may contain credentials or prompts.
                logger.warning(
                    "Could not enrich execution trace for %s", type(event).__name__
                )

    @contextmanager
    def activate(self) -> Iterator[TelemetryExecutionContext]:
        """Bind the session to this context; lifecycle remains with its owner."""
        session_token = _trace_session.set(self)
        context_token = _telemetry_context.set(self.context)
        execution_token = set_execution_uuid(self.context.kickoff_id)
        # Deferred turns leave their root open, but must not leak its event
        # parent into the next execution on the same thread or asyncio task.
        event_scope_token = _event_id_stack.set(_event_id_stack.get())
        try:
            yield self.context
        finally:
            _event_id_stack.reset(event_scope_token)
            clear_execution_uuid(execution_token)
            _telemetry_context.reset(context_token)
            _trace_session.reset(session_token)

    def flush(self, timeout_millis: int = 30000) -> bool:
        return self.tracer_provider.force_flush(timeout_millis)

    def shutdown(self, timeout_millis: int = 30000) -> bool:
        if self._closed:
            return True
        self._closed = True
        for span in self.context.active_spans.values():
            if span.is_recording():
                span.set_status(
                    trace.Status(
                        trace.StatusCode.ERROR, "Execution ended before span completion"
                    )
                )
                span.end()
        self.context.active_spans.clear()
        try:
            return self.flush(timeout_millis)
        finally:
            try:
                self.tracer_provider.shutdown()
            finally:
                self.context._span_refs.clear()
                self.context.pending_span_ends.clear()
                self.context.completed_operations.clear()
                self.context.operation_spans.clear()
                self.context.root_span = None
