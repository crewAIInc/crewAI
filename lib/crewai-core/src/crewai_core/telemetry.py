"""Anonymous telemetry collection — base implementation.

This module is the leaf telemetry layer used by both ``crewai`` (which extends
it with framework-specific spans + event-bus signal hooks) and ``crewai-cli``
(which uses it directly to emit deployment / template / flow-creation spans).

No prompts, task descriptions, agent backstories/goals, responses, or sensitive
data are collected.
"""

from __future__ import annotations

import asyncio
import atexit
from collections.abc import Callable
import contextlib
import logging
import os
import threading
from typing import Any, Final

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExportResult,
)
from opentelemetry.trace import Span, Status, StatusCode
from typing_extensions import Self


logger = logging.getLogger(__name__)


CREWAI_TELEMETRY_BASE_URL: Final[str] = "https://telemetry.crewai.com:4319"
CREWAI_TELEMETRY_SERVICE_NAME: Final[str] = "crewAI-telemetry"


def close_span(span: Span) -> None:
    """Set span status to OK and end it."""
    span.set_status(Status(StatusCode.OK))
    span.end()


@contextlib.contextmanager
def suppress_warnings() -> Any:
    """Suppress noisy warnings during otel provider setup."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


class SafeOTLPSpanExporter(OTLPSpanExporter):
    """OTLP exporter that swallows export failures so telemetry never crashes the app."""

    def export(self, spans: Any) -> SpanExportResult:
        try:
            return super().export(spans)
        except Exception as e:
            logger.debug("Telemetry export failed: %s", e)
            return SpanExportResult.FAILURE


class Telemetry:
    """Base telemetry: OTLP setup + the spans needed by the CLI.

    crewai's runtime extends this with crew/agent/task/tool/flow execution spans
    and event-bus signal handlers (see ``crewai.telemetry.telemetry``).
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls) -> Self:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.ready: bool = False
        self.trace_set: bool = False
        self._initialized: bool = True

        if self._is_telemetry_disabled():
            return

        try:
            self.resource = Resource(
                attributes={SERVICE_NAME: CREWAI_TELEMETRY_SERVICE_NAME},
            )
            with suppress_warnings():
                self.provider = TracerProvider(resource=self.resource)

            processor = BatchSpanProcessor(
                SafeOTLPSpanExporter(
                    endpoint=f"{CREWAI_TELEMETRY_BASE_URL}/v1/traces",
                    timeout=30,
                )
            )

            self.provider.add_span_processor(processor)
            self._register_shutdown_handlers()
            self.ready = True
        except Exception as e:
            if isinstance(
                e,
                (SystemExit, KeyboardInterrupt, GeneratorExit, asyncio.CancelledError),
            ):
                raise
            self.ready = False

    @classmethod
    def _is_telemetry_disabled(cls) -> bool:
        return (
            os.getenv("OTEL_SDK_DISABLED", "false").lower() == "true"
            or os.getenv("CREWAI_DISABLE_TELEMETRY", "false").lower() == "true"
            or os.getenv("CREWAI_DISABLE_TRACKING", "false").lower() == "true"
        )

    def _should_execute_telemetry(self) -> bool:
        return self.ready and not self._is_telemetry_disabled()

    def _register_shutdown_handlers(self) -> None:
        """Register an atexit flush. Subclasses may extend with signal hooks."""
        atexit.register(self._shutdown)

    def _shutdown(self) -> None:
        if not self.ready:
            return
        try:
            self.provider.force_flush(timeout_millis=5000)
            self.provider.shutdown()
            self.ready = False
        except Exception as e:
            logger.debug("Telemetry shutdown failed: %s", e)

    def set_tracer(self) -> None:
        """Install our TracerProvider as the global one (idempotent)."""
        if self.ready and not self.trace_set:
            try:
                with suppress_warnings():
                    trace.set_tracer_provider(self.provider)
                    self.trace_set = True
            except Exception as e:
                logger.debug("Failed to set tracer provider: %s", e)
                self.ready = False
                self.trace_set = False

    def _safe_telemetry_operation(
        self, operation: Callable[[], Span | None]
    ) -> Span | None:
        """Run a span-returning telemetry operation, swallowing failures."""
        if not self._should_execute_telemetry():
            return None
        try:
            return operation()
        except Exception as e:
            logger.debug("Telemetry operation failed: %s", e)
            return None

    def _safe_telemetry_procedure(self, operation: Callable[[], None]) -> None:
        """Run a void telemetry procedure, swallowing failures."""
        if not self._should_execute_telemetry():
            return
        try:
            operation()
        except Exception as e:
            logger.debug("Telemetry operation failed: %s", e)

    def _add_attribute(self, span: Span | None, key: str, value: Any) -> None:
        if span is None:
            return

        def _operation() -> None:
            span.set_attribute(key, value)

        self._safe_telemetry_procedure(_operation)

    # --- CLI-facing spans ---------------------------------------------------

    def deploy_signup_error_span(self) -> None:
        """Records when an error occurs during the deployment signup process."""

        def _operation() -> None:
            tracer = trace.get_tracer("crewai.telemetry")
            span = tracer.start_span("Deploy Signup Error")
            close_span(span)

        self._safe_telemetry_procedure(_operation)

    def start_deployment_span(self, uuid: str | None = None) -> None:
        """Records the start of a deployment process."""

        def _operation() -> None:
            tracer = trace.get_tracer("crewai.telemetry")
            span = tracer.start_span("Start Deployment")
            if uuid:
                self._add_attribute(span, "uuid", uuid)
            close_span(span)

        self._safe_telemetry_procedure(_operation)

    def create_crew_deployment_span(self) -> None:
        """Records the creation of a new crew deployment."""

        def _operation() -> None:
            tracer = trace.get_tracer("crewai.telemetry")
            span = tracer.start_span("Create Crew Deployment")
            close_span(span)

        self._safe_telemetry_procedure(_operation)

    def get_crew_logs_span(
        self, uuid: str | None, log_type: str = "deployment"
    ) -> None:
        """Records the retrieval of crew logs."""

        def _operation() -> None:
            tracer = trace.get_tracer("crewai.telemetry")
            span = tracer.start_span("Get Crew Logs")
            self._add_attribute(span, "log_type", log_type)
            if uuid:
                self._add_attribute(span, "uuid", uuid)
            close_span(span)

        self._safe_telemetry_procedure(_operation)

    def remove_crew_span(self, uuid: str | None = None) -> None:
        """Records the removal of a crew."""

        def _operation() -> None:
            tracer = trace.get_tracer("crewai.telemetry")
            span = tracer.start_span("Remove Crew")
            if uuid:
                self._add_attribute(span, "uuid", uuid)
            close_span(span)

        self._safe_telemetry_procedure(_operation)

    def flow_creation_span(self, flow_name: str) -> None:
        """Records the creation of a new flow."""

        def _operation() -> None:
            tracer = trace.get_tracer("crewai.telemetry")
            span = tracer.start_span("Flow Creation")
            self._add_attribute(span, "flow_name", flow_name)
            close_span(span)

        self._safe_telemetry_procedure(_operation)

    def template_installed_span(self, template_name: str) -> None:
        """Records when a template is downloaded and installed."""
        from crewai_core.version import get_crewai_version

        def _operation() -> None:
            tracer = trace.get_tracer("crewai.telemetry")
            span = tracer.start_span("Template Installed")
            self._add_attribute(span, "crewai_version", get_crewai_version())
            self._add_attribute(span, "template_name", template_name)
            close_span(span)

        self._safe_telemetry_procedure(_operation)
