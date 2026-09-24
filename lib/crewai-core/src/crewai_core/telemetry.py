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
from crewai_core.runtime_env import (
    detect_coding_agent,
    detect_cpu_band,
    detect_runtime_context,
)


logger = logging.getLogger(__name__)


CREWAI_TELEMETRY_BASE_URL: Final[str] = "https://telemetry.crewai.com:4319"
CREWAI_TELEMETRY_SERVICE_NAME: Final[str] = "crewAI-telemetry"

TRACER_NAME: Final[str] = "crewai.telemetry"

DeploySource = Literal["cli", "tui"]
"""Where a deployment was initiated from: a direct CLI command, or the run TUI."""

DeployFailureReason = Literal[
    "api_4xx",
    "api_5xx",
    "invalid_json",
    "invalid_creation_response",
    "network_error",
    "zip_error",
    "user_declined",
    "unexpected",
]
"""Why ``crewai deploy create`` failed after the attempt was counted.

A closed vocabulary, so the warehouse can group on it. ``api_4xx`` / ``api_5xx``
classify the Enterprise API's response (the exact code rides separately as
``status_code``); ``invalid_json`` is a 2xx whose body is not JSON, such as a
proxy's HTML page; ``invalid_creation_response`` a 2xx JSON body that is not a
creation payload (no ``uuid``), which is a broken API contract rather than a
broken network; ``network_error`` a transport failure before any response;
``zip_error`` a failure building the project archive; ``user_declined`` an
abort at a confirmation prompt; ``unexpected`` anything else. Never the error
message.
"""


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


class _ExportState(threading.local):
    """Per-thread marker: True while CrewAI's own exporter runs on this thread."""

    active: bool = False


_export_state = _ExportState()


class _OwnExportLogFilter(logging.Filter):
    """Drop OTLP export logs emitted while CrewAI's own exporter is running.

    ``OTLPSpanExporter`` retries an unreachable collector and logs a warning per
    attempt plus a final error, all on the batch worker thread. That output
    lands on the user's console although the failure is harmless. Scoping the
    filter to that thread keeps the logs of any OTLP exporter the user runs in
    the same process.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return not _export_state.active


_OWN_EXPORT_LOG_FILTER = _OwnExportLogFilter()


class SafeOTLPSpanExporter(OTLPSpanExporter):
    """OTLP exporter that neither raises nor logs when the collector is unreachable."""

    def __init__(self, endpoint: str, timeout: int) -> None:
        super().__init__(endpoint=endpoint, timeout=timeout)
        # Idempotent: a logger holds at most one reference to a given filter.
        logging.getLogger(OTLPSpanExporter.__module__).addFilter(_OWN_EXPORT_LOG_FILTER)

    def export(self, spans: Any) -> SpanExportResult:
        _export_state.active = True
        try:
            return super().export(spans)
        except Exception as e:
            logger.debug("Telemetry export failed: %s", e)
            return SpanExportResult.FAILURE
        finally:
            _export_state.active = False

    def shutdown(self) -> None:
        # flush_and_shutdown stops the exporter before the processor does, and
        # the base class logs a warning for the repeat call.
        _export_state.active = True
        try:
            super().shutdown()  # type: ignore[no-untyped-call]  # unannotated upstream
        finally:
            _export_state.active = False


FINAL_FLUSH_SECONDS: Final[int] = 10


def flush_and_shutdown(
    provider: TracerProvider, exporter: SafeOTLPSpanExporter
) -> None:
    """Export what is still buffered, waiting at most ``FINAL_FLUSH_SECONDS``.

    ``BatchSpanProcessor.force_flush`` ignores its timeout and runs the export,
    retry loop included, on the calling thread
    (open-telemetry/opentelemetry-python#4568). With the collector unreachable
    that held process exit for the exporter's whole retry budget. Flushing on a
    helper thread and then stopping the exporter ends the loop at the deadline;
    when the export succeeds sooner, the join returns as soon as it is done.
    """
    flush = threading.Thread(
        target=provider.force_flush, name="crewai-telemetry-flush", daemon=True
    )
    flush.start()
    flush.join(FINAL_FLUSH_SECONDS)
    exporter.shutdown()
    provider.shutdown()


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
        Attributes to stamp on every span. ``project_id`` is always present and
        is the empty string whenever no id is available -- both for projects that
        declare none and when the lookup itself failed, which are deliberately
        indistinguishable here because neither yields an id. See the comment at
        the assignment for why empty and *absent* must stay distinct.
    """
    attributes = {
        "coding_agent": detect_coding_agent(),
        "runtime_context": detect_runtime_context(),
        "cpu_band": detect_cpu_band(),
    }

    try:
        # Read-only: minting an id belongs to the CLI commands a user
        # invoked, not to a library call during execution.
        project_id = get_project_id()
    except Exception as e:  # Telemetry must never break execution.
        logger.debug("Failed to read project id: %s", e)
        project_id = None

    # Always set the key, even when empty. Absent and empty mean different things
    # and only this distinction can tell them apart: absent means the client is too
    # old to report a project id at all, empty means the client asked and the project
    # declares none -- or the lookup failed, which lands here too and is treated the
    # same, since an unreadable pyproject.toml also means no id is available.
    # Collapsing empty into "absent" makes the share of clients that COULD have
    # reported one unknowable, and that share is the denominator every attribution
    # rate needs.
    attributes["project_id"] = project_id or ""

    return attributes


class Telemetry:
    """Base telemetry: OTLP setup + the spans needed by the CLI.

    crewai's runtime extends this with crew/agent/task/tool/flow execution spans
    and event-bus signal handlers (see ``crewai.telemetry.telemetry``).
    """

    _instance: ClassVar[Self | None] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()
    _TRUTHY_ENV: ClassVar[frozenset[str]] = frozenset({"1", "on", "true", "yes"})
    _FALSY_ENV: ClassVar[frozenset[str]] = frozenset({"", "0", "false", "no", "off"})
    _warned_env_flags: ClassVar[set[tuple[str, str]]] = set()

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

            self._exporter = SafeOTLPSpanExporter(
                endpoint=f"{CREWAI_TELEMETRY_BASE_URL}/v1/traces",
                timeout=30,
            )
            self.provider.add_span_processor(BatchSpanProcessor(self._exporter))
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
    def _env_flag_enabled(cls, name: str, *, default: bool = False) -> bool:
        """Return whether ``name`` is a conventional yes-value.

        Yes: ``true``, ``1``, ``yes``, ``on``. No: unset, ``false``, ``0``,
        ``no``, ``off``, empty. Anything else is treated as unset and logged
        once per ``(name, raw)`` pair for the process.
        """
        raw = os.getenv(name)
        if raw is None:
            return default
        value = raw.strip().lower()
        if value in cls._TRUTHY_ENV:
            return True
        if value in cls._FALSY_ENV:
            return False
        warning_key = (name, raw)
        if warning_key not in cls._warned_env_flags:
            cls._warned_env_flags.add(warning_key)
            logger.warning(
                "Unrecognized value %r for %s; expected true/1/yes/on or "
                "false/0/no/off. Treating as unset.",
                raw,
                name,
            )
        return default

    @classmethod
    def _is_telemetry_disabled(cls) -> bool:
        return (
            cls._env_flag_enabled("OTEL_SDK_DISABLED")
            or cls._env_flag_enabled("CREWAI_DISABLE_TELEMETRY")
            or cls._env_flag_enabled("CREWAI_DISABLE_TRACKING")
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
            flush_and_shutdown(self.provider, self._exporter)
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

    def crew_deployment_created_span(
        self, uuid: str | None = None, source: DeploySource = "cli"
    ) -> None:
        """Records that a crew deployment was confirmed created, with its uuid.

        Distinct from :meth:`create_crew_deployment_span`, which fires *before*
        the API call and so counts creation **attempts**. The uuid cannot be on
        that span: the call that creates the deployment is the call that returns
        the uuid, so it does not exist yet. Attribution therefore needs a second
        span, emitted once the response has validated.

        Emits no feature count on purpose. ``create_crew_deployment_span``
        already emits ``deploy:created``; a second emit would double the
        deployment count that origin-independent aggregation depends on.

        Args:
            uuid: The deployment that was created.
            source: Where the deployment was initiated from.
        """

        from crewai_core.version import get_crewai_version

        def _operation() -> None:
            tracer = self.provider.get_tracer(TRACER_NAME)
            span = tracer.start_span("Crew Deployment Created")
            self._add_attribute(span, "crewai_version", get_crewai_version())
            if uuid:
                self._add_attribute(span, "uuid", uuid)
            self._add_attribute(span, "source", source)
            close_span(span)

        self._safe_telemetry_procedure(_operation)

    def crew_deployment_failed_span(
        self,
        reason: DeployFailureReason,
        source: DeploySource = "cli",
        status_code: int | None = None,
    ) -> None:
        """Records that ``crewai deploy create`` failed after the attempt was counted.

        :meth:`create_crew_deployment_span` counts attempts and
        :meth:`crew_deployment_created_span` counts successes; the gap between
        them was measurable but had no cause attached. This span carries the
        cause from a closed vocabulary, plus the HTTP status when the API
        answered. Emits no feature count, for the same reason as
        :meth:`crew_deployment_created_span`.

        Args:
            reason: Why the create failed.
            source: Where the deployment was initiated from.
            status_code: HTTP status of the API response, when there was one.
        """

        from crewai_core.version import get_crewai_version

        def _operation() -> None:
            tracer = self.provider.get_tracer(TRACER_NAME)
            span = tracer.start_span("Crew Deployment Failed")
            self._add_attribute(span, "crewai_version", get_crewai_version())
            self._add_attribute(span, "reason", reason)
            if status_code is not None:
                self._add_attribute(span, "status_code", status_code)
            self._add_attribute(span, "source", source)
            close_span(span)

        self._safe_telemetry_procedure(_operation)

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

    def project_created_span(self, kind: str, project_id: str | None) -> None:
        """Records that the CLI scaffolded a new project.

        Acquisition was previously only observable from a project's first *run*, which
        misses every project created and never run and dates the rest to the wrong day.

        ``created_project_id`` rather than ``project_id``: the ``project_id`` stamped on
        every span by ``CommonAttributesSpanProcessor`` is read from the *current
        working directory* and cached for the life of the process, so at scaffold time it
        describes the directory the command was run from - not the project just minted a
        line earlier. Two different things must not share one attribute name.

        Args:
            kind: What was scaffolded - "crew", "json_crew" or "flow".
            project_id: The id just minted for the new project. Empty string when
                minting failed, matching the convention for the common attribute.
        """
        from crewai_core.version import get_crewai_version

        def _operation() -> None:
            tracer = self.provider.get_tracer(TRACER_NAME)
            span = tracer.start_span("Project Created")
            self._add_attribute(span, "crewai_version", get_crewai_version())
            self._add_attribute(span, "kind", kind)
            self._add_attribute(span, "created_project_id", project_id or "")
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
