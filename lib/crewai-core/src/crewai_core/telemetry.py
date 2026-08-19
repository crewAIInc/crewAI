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
from functools import cache
import logging
import os
import threading
from typing import Any, ClassVar, Final, Literal

from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExportResult,
)
from opentelemetry.trace import Span, Status, StatusCode
from typing_extensions import Self

from crewai_core.project import get_project_id
from crewai_core.runtime_env import detect_coding_agent, detect_runtime_context


logger = logging.getLogger(__name__)


CREWAI_TELEMETRY_BASE_URL: Final[str] = "https://telemetry.crewai.com:4319"
CREWAI_TELEMETRY_SERVICE_NAME: Final[str] = "crewAI-telemetry"

TRACER_NAME: Final[str] = "crewai.telemetry"

DeploySource = Literal["cli", "tui"]
"""Where a deployment was initiated from: a direct CLI command, or the run TUI."""


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


class CommonAttributesSpanProcessor(SpanProcessor):
    """Applies a fixed set of attributes to every span at start.

    Used for process-wide context that should appear on all spans (e.g. which
    AI coding assistant is running the process) without each span-emitting
    method having to set it. Attributes are applied as span attributes rather
    than Resource attributes because the ingestion pipeline preserves only
    serviceName from the resource.
    """

    def __init__(self, attributes: dict[str, str]) -> None:
        """Initialize the processor.

        Args:
            attributes: Attributes applied to every span. Values must not
                contain user data - this is process-wide context only.
        """
        self._attributes = attributes

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        """Apply the common attributes to a span as it starts.

        Args:
            span: The span being started.
            parent_context: Parent context, unused.
        """
        try:
            span.set_attributes(self._attributes)
        except Exception:  # noqa: S110 - telemetry must never break execution
            pass

    def on_end(self, span: Any) -> None:
        """No-op; export is handled by the batch processor."""

    def shutdown(self) -> None:
        """No-op; this processor holds no resources."""

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """No-op flush.

        Args:
            timeout_millis: Unused.

        Returns:
            Always True.
        """
        return True


@cache
def common_span_attributes() -> dict[str, str]:
    """Build the attributes every span carries, once per process.

    Cached because it reads the project's ``pyproject.toml``, and because a
    process cannot change which assistant, runtime, or project it belongs to
    partway through.

    Shared by both telemetry implementations so a CLI-only process, which never
    imports ``crewai``, still reports the same process-wide context.

    Returns:
        Attributes to stamp on every span. ``project_id`` is omitted for
        projects that do not declare one.
    """
    attributes = {
        "coding_agent": detect_coding_agent(),
        "runtime_context": detect_runtime_context(),
    }

    try:
        # Read-only: minting an id belongs to the CLI commands a user
        # invoked, not to a library call during execution.
        project_id = get_project_id()
    except Exception as e:  # Telemetry must never break execution.
        logger.debug("Failed to read project id: %s", e)
        project_id = None

    if project_id:
        attributes["project_id"] = project_id

    return attributes


class Telemetry:
    """Base telemetry: OTLP setup + the spans needed by the CLI.

    crewai's runtime extends this with crew/agent/task/tool/flow execution spans
    and event-bus signal handlers (see ``crewai.telemetry.telemetry``).
    """

    _instance: ClassVar[Self | None] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

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

            # Span attributes, not Resource attributes: the ingestion pipeline
            # preserves only serviceName from the resource, so anything else set
            # there is dropped before it reaches storage.
            self.provider.add_span_processor(
                CommonAttributesSpanProcessor(common_span_attributes())
            )

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
        """Mark telemetry live (idempotent).

        Deliberately does not install a global TracerProvider. Doing so made
        every OTel-instrumented library in the host process - HTTP servers,
        Redis clients, ORMs - resolve ``trace.get_tracer()`` to our provider
        and export to our collector. It also meant that when an application
        had already installed its own provider, our spans went to *their*
        collector instead of ours. Spans are now created from ``self.provider``
        directly, so neither can happen.

        Retained rather than removed because it is called across packages
        (``crewai_cli.command``, ``crewai_cli.crew_run_tui``) and by the
        ``crewai`` event listener.
        """
        self.trace_set = self.ready

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

    def deploy_signup_error_span(self) -> None:
        """Records when an error occurs during the deployment signup process."""

        from crewai_core.version import get_crewai_version

        def _operation() -> None:
            tracer = self.provider.get_tracer(TRACER_NAME)
            span = tracer.start_span("Deploy Signup Error")
            self._add_attribute(span, "crewai_version", get_crewai_version())
            close_span(span)

        self._safe_telemetry_procedure(_operation)

    def start_deployment_span(
        self, uuid: str | None = None, source: DeploySource = "cli"
    ) -> None:
        """Records redeploying an existing crew (``crewai deploy push``).

        Also emits ``deploy:pushed`` so that deployments are countable from the
        feature-usage aggregation regardless of where they were started from.

        Args:
            uuid: The deployment being pushed to.
            source: Where the deployment was initiated from.
        """

        from crewai_core.version import get_crewai_version

        def _operation() -> None:
            tracer = self.provider.get_tracer(TRACER_NAME)
            span = tracer.start_span("Start Deployment")
            self._add_attribute(span, "crewai_version", get_crewai_version())
            if uuid:
                self._add_attribute(span, "uuid", uuid)
            self._add_attribute(span, "source", source)
            close_span(span)

        self._safe_telemetry_procedure(_operation)
        self.feature_usage_span("deploy:pushed")

    def create_crew_deployment_span(self, source: DeploySource = "cli") -> None:
        """Records creating a new crew deployment (``crewai deploy create``).

        Also emits ``deploy:created`` so that deployments are countable from the
        feature-usage aggregation regardless of where they were started from.

        Args:
            source: Where the deployment was initiated from.
        """

        from crewai_core.version import get_crewai_version

        def _operation() -> None:
            tracer = self.provider.get_tracer(TRACER_NAME)
            span = tracer.start_span("Create Crew Deployment")
            self._add_attribute(span, "crewai_version", get_crewai_version())
            self._add_attribute(span, "source", source)
            close_span(span)

        self._safe_telemetry_procedure(_operation)
        self.feature_usage_span("deploy:created")

    def get_crew_logs_span(
        self, uuid: str | None, log_type: str = "deployment"
    ) -> None:
        """Records the retrieval of crew logs."""

        from crewai_core.version import get_crewai_version

        def _operation() -> None:
            tracer = self.provider.get_tracer(TRACER_NAME)
            span = tracer.start_span("Get Crew Logs")
            self._add_attribute(span, "crewai_version", get_crewai_version())
            self._add_attribute(span, "log_type", log_type)
            if uuid:
                self._add_attribute(span, "uuid", uuid)
            close_span(span)

        self._safe_telemetry_procedure(_operation)

    def remove_crew_span(self, uuid: str | None = None) -> None:
        """Records the removal of a crew."""

        from crewai_core.version import get_crewai_version

        def _operation() -> None:
            tracer = self.provider.get_tracer(TRACER_NAME)
            span = tracer.start_span("Remove Crew")
            self._add_attribute(span, "crewai_version", get_crewai_version())
            if uuid:
                self._add_attribute(span, "uuid", uuid)
            close_span(span)

        self._safe_telemetry_procedure(_operation)

    def feature_usage_span(self, feature: str) -> None:
        """Records that a feature was used. One span = one count."""
        from crewai_core.version import get_crewai_version

        def _operation() -> None:
            tracer = self.provider.get_tracer(TRACER_NAME)
            span = tracer.start_span("Feature Usage")
            self._add_attribute(span, "crewai_version", get_crewai_version())
            self._add_attribute(span, "feature", feature)
            close_span(span)

        self._safe_telemetry_procedure(_operation)

    def flow_creation_span(self, flow_name: str) -> None:
        """Records the creation of a new flow."""
        from crewai_core.version import get_crewai_version

        def _operation() -> None:
            tracer = self.provider.get_tracer(TRACER_NAME)
            span = tracer.start_span("Flow Creation")
            self._add_attribute(span, "crewai_version", get_crewai_version())
            self._add_attribute(span, "flow_name", flow_name)
            close_span(span)

        self._safe_telemetry_procedure(_operation)

    def template_installed_span(self, template_name: str) -> None:
        """Records when a template is downloaded and installed."""
        from crewai_core.version import get_crewai_version

        def _operation() -> None:
            tracer = self.provider.get_tracer(TRACER_NAME)
            span = tracer.start_span("Template Installed")
            self._add_attribute(span, "crewai_version", get_crewai_version())
            self._add_attribute(span, "template_name", template_name)
            close_span(span)

        self._safe_telemetry_procedure(_operation)
