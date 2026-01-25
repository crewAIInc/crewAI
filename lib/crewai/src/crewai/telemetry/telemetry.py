"""Telemetry module for CrewAI.

This module provides anonymous telemetry collection for development purposes.
No prompts, task descriptions, agent backstories/goals, responses, or sensitive
data is collected. Users can opt-in to share more complete data using the
`share_crew` attribute.
"""

from __future__ import annotations

import asyncio
import atexit
from collections.abc import Callable
from importlib.metadata import version
import json
import logging
import os
import platform
import signal
import threading
from typing import TYPE_CHECKING, Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExportResult,
)
from opentelemetry.trace import Span
from typing_extensions import Self

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.system_events import (
    SigContEvent,
    SigHupEvent,
    SigIntEvent,
    SigTStpEvent,
    SigTermEvent,
)
from crewai.telemetry.constants import (
    CREWAI_TELEMETRY_BASE_URL,
    CREWAI_TELEMETRY_SERVICE_NAME,
)
from crewai.telemetry.utils import (
    add_agent_fingerprint_to_span,
    add_crew_and_task_attributes,
    add_crew_attributes,
    close_span,
)
from crewai.utilities.logger_utils import suppress_warnings
from crewai.utilities.string_utils import sanitize_tool_name


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from crewai.crew import Crew
    from crewai.task import Task


class SafeOTLPSpanExporter(OTLPSpanExporter):
    """Safe wrapper for OTLP span exporter that handles exceptions gracefully.

    This exporter prevents telemetry failures from breaking the application
    by catching and logging exceptions during span export.
    """

    def export(self, spans: Any) -> SpanExportResult:
        """Export spans to the telemetry backend safely.

        Args:
            spans: Collection of spans to export.

        Returns:
            Export result status, FAILURE if an exception occurs.
        """
        try:
            return super().export(spans)
        except Exception as e:
            logger.error(e)
            return SpanExportResult.FAILURE


class Telemetry:
    """Handle anonymous telemetry for the CrewAI package.

    Attributes:
        ready: Whether telemetry is initialized and ready.
        trace_set: Whether the tracer provider has been set.
        resource: OpenTelemetry resource for the telemetry service.
        provider: OpenTelemetry tracer provider.
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
                raise  # Re-raise the exception to not interfere with system signals
            self.ready = False

    @classmethod
    def _is_telemetry_disabled(cls) -> bool:
        """Check if telemetry should be disabled based on environment variables."""
        return (
            os.getenv("OTEL_SDK_DISABLED", "false").lower() == "true"
            or os.getenv("CREWAI_DISABLE_TELEMETRY", "false").lower() == "true"
            or os.getenv("CREWAI_DISABLE_TRACKING", "false").lower() == "true"
        )

    def _should_execute_telemetry(self) -> bool:
        """Check if telemetry operations should be executed."""
        return self.ready and not self._is_telemetry_disabled()

    def set_tracer(self) -> None:
        """Set the tracer provider if ready and not already set."""
        if self.ready and not self.trace_set:
            try:
                with suppress_warnings():
                    trace.set_tracer_provider(self.provider)
                    self.trace_set = True
            except Exception as e:
                logger.debug(f"Failed to set tracer provider: {e}")
                self.ready = False
                self.trace_set = False

    def _register_shutdown_handlers(self) -> None:
        """Register handlers for graceful shutdown on process exit and signals."""
        atexit.register(self._shutdown)

        self._original_handlers: dict[int, Any] = {}

        self._register_signal_handler(signal.SIGTERM, SigTermEvent, shutdown=True)
        self._register_signal_handler(signal.SIGINT, SigIntEvent, shutdown=True)
        if hasattr(signal, "SIGHUP"):
            self._register_signal_handler(signal.SIGHUP, SigHupEvent, shutdown=False)
        if hasattr(signal, "SIGTSTP"):
            self._register_signal_handler(signal.SIGTSTP, SigTStpEvent, shutdown=False)
        if hasattr(signal, "SIGCONT"):
            self._register_signal_handler(signal.SIGCONT, SigContEvent, shutdown=False)

    def _register_signal_handler(
        self,
        sig: signal.Signals,
        event_class: type,
        shutdown: bool = False,
    ) -> None:
        """Register a signal handler that emits an event.

        Args:
            sig: The signal to handle.
            event_class: The event class to instantiate and emit.
            shutdown: Whether to trigger shutdown on this signal.
        """
        try:
            original_handler = signal.getsignal(sig)
            self._original_handlers[sig] = original_handler

            def handler(signum: int, frame: Any) -> None:
                crewai_event_bus.emit(self, event_class())

                if shutdown:
                    self._shutdown()

                if original_handler not in (signal.SIG_DFL, signal.SIG_IGN, None):
                    if callable(original_handler):
                        original_handler(signum, frame)
                elif shutdown:
                    raise SystemExit(0)

            signal.signal(sig, handler)
        except ValueError as e:
            logger.warning(
                f"Cannot register {sig.name} handler: not running in main thread",
                exc_info=e,
            )
        except OSError as e:
            logger.warning(f"Cannot register {sig.name} handler: {e}", exc_info=e)

    def _shutdown(self) -> None:
        """Flush and shutdown the telemetry provider on process exit.

        Uses a short timeout to avoid blocking process shutdown.
        """
        if not self.ready:
            return

        try:
            self.provider.force_flush(timeout_millis=5000)
            self.provider.shutdown()
            self.ready = False
        except Exception as e:
            logger.debug(f"Telemetry shutdown failed: {e}")

    def _safe_telemetry_operation(
        self, operation: Callable[[], Span | None]
    ) -> Span | None:
        """Execute telemetry operation safely, checking both readiness and environment variables.

        Args:
            operation: A callable that performs telemetry operations.

        Returns:
            The return value from the operation, or None if telemetry is disabled or fails.
        """
        if not self._should_execute_telemetry():
            return None
        try:
            return operation()
        except Exception as e:
            logger.debug(f"Telemetry operation failed: {e}")
            return None

    def crew_creation(self, crew: Crew, inputs: dict[str, Any] | None) -> None:
        """Records the creation of a crew.

        Args:
            crew: The crew being created.
            inputs: Optional input parameters for the crew.
        """

        def _operation() -> None:
            tracer = trace.get_tracer("crewai.telemetry")
            span = tracer.start_span("Crew Created")
            self._add_attribute(
                span,
                "crewai_version",
                version("crewai"),
            )
            self._add_attribute(span, "python_version", platform.python_version())
            add_crew_attributes(span, crew, self._add_attribute)
            self._add_attribute(span, "crew_process", crew.process)
            self._add_attribute(span, "crew_memory", crew.memory)
            self._add_attribute(span, "crew_number_of_tasks", len(crew.tasks))
            self._add_attribute(span, "crew_number_of_agents", len(crew.agents))

            # Add additional fingerprint metadata if available
            if hasattr(crew, "fingerprint") and crew.fingerprint:
                self._add_attribute(
                    span,
                    "crew_fingerprint_created_at",
                    crew.fingerprint.created_at.isoformat(),
                )
                # Add fingerprint metadata if it exists
                if hasattr(crew.fingerprint, "metadata") and crew.fingerprint.metadata:
                    self._add_attribute(
                        span,
                        "crew_fingerprint_metadata",
                        json.dumps(crew.fingerprint.metadata),
                    )

            if crew.share_crew:
                self._add_attribute(
                    span,
                    "crew_agents",
                    json.dumps(
                        [
                            {
                                "key": agent.key,
                                "id": str(agent.id),
                                "role": agent.role,
                                "goal": agent.goal,
                                "backstory": agent.backstory,
                                "verbose?": agent.verbose,
                                "max_iter": agent.max_iter,
                                "max_rpm": agent.max_rpm,
                                "i18n": agent.i18n.prompt_file,
                                "function_calling_llm": (
                                    getattr(
                                        getattr(agent, "function_calling_llm", None),
                                        "model",
                                        "",
                                    )
                                    if getattr(agent, "function_calling_llm", None)
                                    else ""
                                ),
                                "llm": agent.llm.model,
                                "delegation_enabled?": agent.allow_delegation,
                                "allow_code_execution?": getattr(
                                    agent, "allow_code_execution", False
                                ),
                                "max_retry_limit": getattr(agent, "max_retry_limit", 3),
                                "tools_names": [
                                    sanitize_tool_name(tool.name)
                                    for tool in agent.tools or []
                                ],
                                # Add agent fingerprint data if sharing crew details
                                "fingerprint": (
                                    getattr(
                                        getattr(agent, "fingerprint", None),
                                        "uuid_str",
                                        None,
                                    )
                                ),
                                "fingerprint_created_at": (
                                    created_at.isoformat()
                                    if (
                                        created_at := getattr(
                                            getattr(agent, "fingerprint", None),
                                            "created_at",
                                            None,
                                        )
                                    )
                                    is not None
                                    else None
                                ),
                            }
                            for agent in crew.agents
                        ]
                    ),
                )
                self._add_attribute(
                    span,
                    "crew_tasks",
                    json.dumps(
                        [
                            {
                                "key": task.key,
                                "id": str(task.id),
                                "description": task.description,
                                "expected_output": task.expected_output,
                                "async_execution?": task.async_execution,
                                "human_input?": task.human_input,
                                "agent_role": (
                                    task.agent.role if task.agent else "None"
                                ),
                                "agent_key": task.agent.key if task.agent else None,
                                "context": (
                                    [task.description for task in task.context]
                                    if isinstance(task.context, list)
                                    else None
                                ),
                                "tools_names": [
                                    sanitize_tool_name(tool.name)
                                    for tool in task.tools or []
                                ],
                                # Add task fingerprint data if sharing crew details
                                "fingerprint": (
                                    task.fingerprint.uuid_str
                                    if hasattr(task, "fingerprint") and task.fingerprint
                                    else None
                                ),
                                "fingerprint_created_at": (
                                    task.fingerprint.created_at.isoformat()
                                    if hasattr(task, "fingerprint") and task.fingerprint
                                    else None
                                ),
                            }
                            for task in crew.tasks
                        ]
                    ),
                )
                self._add_attribute(span, "platform", platform.platform())
                self._add_attribute(span, "platform_release", platform.release())
                self._add_attribute(span, "platform_system", platform.system())
                self._add_attribute(span, "platform_version", platform.version())
                self._add_attribute(span, "cpus", os.cpu_count())
                self._add_attribute(span, "crew_inputs", json.dumps(inputs or {}))
            else:
                self._add_attribute(
                    span,
                    "crew_agents",
                    json.dumps(
                        [
                            {
                                "key": agent.key,
                                "id": str(agent.id),
                                "role": agent.role,
                                "verbose?": agent.verbose,
                                "max_iter": agent.max_iter,
                                "max_rpm": agent.max_rpm,
                                "function_calling_llm": (
                                    getattr(
                                        getattr(agent, "function_calling_llm", None),
                                        "model",
                                        "",
                                    )
                                    if getattr(agent, "function_calling_llm", None)
                                    else ""
                                ),
                                "llm": agent.llm.model,
                                "delegation_enabled?": agent.allow_delegation,
                                "allow_code_execution?": getattr(
                                    agent, "allow_code_execution", False
                                ),
                                "max_retry_limit": getattr(agent, "max_retry_limit", 3),
                                "tools_names": [
                                    sanitize_tool_name(tool.name)
                                    for tool in agent.tools or []
                                ],
                            }
                            for agent in crew.agents
                        ]
                    ),
                )
                self._add_attribute(
                    span,
                    "crew_tasks",
                    json.dumps(
                        [
                            {
                                "key": task.key,
                                "id": str(task.id),
                                "async_execution?": task.async_execution,
                                "human_input?": task.human_input,
                                "agent_role": (
                                    task.agent.role if task.agent else "None"
                                ),
                                "agent_key": task.agent.key if task.agent else None,
                                "tools_names": [
                                    sanitize_tool_name(tool.name)
                                    for tool in task.tools or []
                                ],
                            }
                            for task in crew.tasks
                        ]
                    ),
                )
            close_span(span)

        self._safe_telemetry_operation(_operation)

    def task_started(self, crew: Crew, task: Task) -> Span | None:
        """Records task started in a crew.

        Args:
            crew: The crew executing the task.
            task: The task being started.

        Returns:
            The span tracking the task execution, or None if telemetry is disabled.
        """

        def _operation() -> Span:
            tracer = trace.get_tracer("crewai.telemetry")

            created_span = tracer.start_span("Task Created")

            add_crew_and_task_attributes(created_span, crew, task, self._add_attribute)

            if hasattr(task, "fingerprint") and task.fingerprint:
                self._add_attribute(
                    created_span, "task_fingerprint", task.fingerprint.uuid_str
                )
                self._add_attribute(
                    created_span,
                    "task_fingerprint_created_at",
                    task.fingerprint.created_at.isoformat(),
                )
                # Add fingerprint metadata if it exists
                if hasattr(task.fingerprint, "metadata") and task.fingerprint.metadata:
                    self._add_attribute(
                        created_span,
                        "task_fingerprint_metadata",
                        json.dumps(task.fingerprint.metadata),
                    )

            # Add agent fingerprint if task has an assigned agent
            if hasattr(task, "agent") and task.agent:
                add_agent_fingerprint_to_span(
                    created_span, task.agent, self._add_attribute
                )

            if crew.share_crew:
                self._add_attribute(
                    created_span, "formatted_description", task.description
                )
                self._add_attribute(
                    created_span, "formatted_expected_output", task.expected_output
                )

            close_span(created_span)

            span = tracer.start_span("Task Execution")

            add_crew_and_task_attributes(span, crew, task, self._add_attribute)

            if hasattr(task, "fingerprint") and task.fingerprint:
                self._add_attribute(span, "task_fingerprint", task.fingerprint.uuid_str)

            # Add agent fingerprint if task has an assigned agent
            if hasattr(task, "agent") and task.agent:
                add_agent_fingerprint_to_span(span, task.agent, self._add_attribute)

            if crew.share_crew:
                self._add_attribute(span, "formatted_description", task.description)
                self._add_attribute(
                    span, "formatted_expected_output", task.expected_output
                )

            return span

        return self._safe_telemetry_operation(_operation)

    def task_ended(self, span: Span, task: Task, crew: Crew) -> None:
        """Records the completion of a task execution in a crew.

        Args:
            span: The OpenTelemetry span tracking the task execution.
            task: The task that was completed.
            crew: The crew context in which the task was executed.

        Note:
            If share_crew is enabled, this will also record the task output.
        """

        def _operation() -> None:
            # Ensure fingerprint data is present on completion span
            if hasattr(task, "fingerprint") and task.fingerprint:
                self._add_attribute(span, "task_fingerprint", task.fingerprint.uuid_str)

            if crew.share_crew:
                self._add_attribute(
                    span,
                    "task_output",
                    task.output.raw if task.output else "",
                )

            close_span(span)

        self._safe_telemetry_operation(_operation)

    def tool_repeated_usage(self, llm: Any, tool_name: str, attempts: int) -> None:
        """Records when a tool is used repeatedly, which might indicate an issue.

        Args:
            llm: The language model being used.
            tool_name: Name of the tool being repeatedly used.
            attempts: Number of attempts made with this tool.
        """

        def _operation() -> None:
            tracer = trace.get_tracer("crewai.telemetry")
            span = tracer.start_span("Tool Repeated Usage")
            self._add_attribute(
                span,
                "crewai_version",
                version("crewai"),
            )
            self._add_attribute(span, "tool_name", tool_name)
            self._add_attribute(span, "attempts", attempts)
            if llm:
                self._add_attribute(span, "llm", llm.model)
            close_span(span)

        self._safe_telemetry_operation(_operation)

    def tool_usage(
        self, llm: Any, tool_name: str, attempts: int, agent: Any = None
    ) -> None:
        """Records the usage of a tool by an agent.

        Args:
            llm: The language model being used.
            tool_name: Name of the tool being used.
            attempts: Number of attempts made with this tool.
            agent: The agent using the tool.
        """

        def _operation() -> None:
            tracer = trace.get_tracer("crewai.telemetry")
            span = tracer.start_span("Tool Usage")
            self._add_attribute(
                span,
                "crewai_version",
                version("crewai"),
            )
            self._add_attribute(span, "tool_name", tool_name)
            self._add_attribute(span, "attempts", attempts)
            if llm:
                self._add_attribute(span, "llm", llm.model)

            # Add agent fingerprint data if available
            add_agent_fingerprint_to_span(span, agent, self._add_attribute)
            close_span(span)

        self._safe_telemetry_operation(_operation)

    def tool_usage_error(
        self, llm: Any, agent: Any = None, tool_name: str | None = None
    ) -> None:
        """Records when a tool usage results in an error.

        Args:
            llm: The language model being used when the error occurred.
            agent: The agent using the tool.
            tool_name: Name of the tool that caused the error.
        """

        def _operation() -> None:
            tracer = trace.get_tracer("crewai.telemetry")
            span = tracer.start_span("Tool Usage Error")
            self._add_attribute(
                span,
                "crewai_version",
                version("crewai"),
            )
            if llm:
                self._add_attribute(span, "llm", llm.model)

            if tool_name:
                self._add_attribute(span, "tool_name", tool_name)

            # Add agent fingerprint data if available
            add_agent_fingerprint_to_span(span, agent, self._add_attribute)
            close_span(span)

        self._safe_telemetry_operation(_operation)

    def individual_test_result_span(
        self, crew: Crew, quality: float, exec_time: int, model_name: str
    ) -> None:
        """Records individual test results for a crew execution.

        Args:
            crew: The crew being tested.
            quality: Quality score of the execution.
            exec_time: Execution time in seconds.
            model_name: Name of the model used.
        """

        def _operation() -> None:
            tracer = trace.get_tracer("crewai.telemetry")
            span = tracer.start_span("Crew Individual Test Result")

            self._add_attribute(
                span,
                "crewai_version",
                version("crewai"),
            )
            add_crew_attributes(
                span, crew, self._add_attribute, include_fingerprint=False
            )
            self._add_attribute(span, "quality", str(quality))
            self._add_attribute(span, "exec_time", str(exec_time))
            self._add_attribute(span, "model_name", model_name)
            close_span(span)

        self._safe_telemetry_operation(_operation)

    def test_execution_span(
        self,
        crew: Crew,
        iterations: int,
        inputs: dict[str, Any] | None,
        model_name: str,
    ) -> None:
        """Records the execution of a test suite for a crew.

        Args:
            crew: The crew being tested.
            iterations: Number of test iterations.
            inputs: Input parameters for the test.
            model_name: Name of the model used in testing.
        """

        def _operation() -> None:
            tracer = trace.get_tracer("crewai.telemetry")
            span = tracer.start_span("Crew Test Execution")

            self._add_attribute(
                span,
                "crewai_version",
                version("crewai"),
            )
            add_crew_attributes(
                span, crew, self._add_attribute, include_fingerprint=False
            )
            self._add_attribute(span, "iterations", str(iterations))
            self._add_attribute(span, "model_name", model_name)

            if crew.share_crew:
                self._add_attribute(span, "inputs", json.dumps(inputs or {}))

            close_span(span)

        self._safe_telemetry_operation(_operation)

    def deploy_signup_error_span(self) -> None:
        """Records when an error occurs during the deployment signup process."""

        def _operation() -> None:
            tracer = trace.get_tracer("crewai.telemetry")
            span = tracer.start_span("Deploy Signup Error")
            close_span(span)

        self._safe_telemetry_operation(_operation)

    def start_deployment_span(self, uuid: str | None = None) -> None:
        """Records the start of a deployment process.

        Args:
            uuid: Unique identifier for the deployment.
        """

        def _operation() -> None:
            tracer = trace.get_tracer("crewai.telemetry")
            span = tracer.start_span("Start Deployment")
            if uuid:
                self._add_attribute(span, "uuid", uuid)
            close_span(span)

        self._safe_telemetry_operation(_operation)

    def create_crew_deployment_span(self) -> None:
        """Records the creation of a new crew deployment."""

        def _operation() -> None:
            tracer = trace.get_tracer("crewai.telemetry")
            span = tracer.start_span("Create Crew Deployment")
            close_span(span)

        self._safe_telemetry_operation(_operation)

    def get_crew_logs_span(
        self, uuid: str | None, log_type: str = "deployment"
    ) -> None:
        """Records the retrieval of crew logs.

        Args:
            uuid: Unique identifier for the crew.
            log_type: Type of logs being retrieved. Defaults to "deployment".
        """

        def _operation() -> None:
            tracer = trace.get_tracer("crewai.telemetry")
            span = tracer.start_span("Get Crew Logs")
            self._add_attribute(span, "log_type", log_type)
            if uuid:
                self._add_attribute(span, "uuid", uuid)
            close_span(span)

        self._safe_telemetry_operation(_operation)

    def remove_crew_span(self, uuid: str | None = None) -> None:
        """Records the removal of a crew.

        Args:
            uuid: Unique identifier for the crew being removed.
        """

        def _operation() -> None:
            tracer = trace.get_tracer("crewai.telemetry")
            span = tracer.start_span("Remove Crew")
            if uuid:
                self._add_attribute(span, "uuid", uuid)
            close_span(span)

        self._safe_telemetry_operation(_operation)

    def crew_execution_span(
        self, crew: Crew, inputs: dict[str, Any] | None
    ) -> Span | None:
        """Records the complete execution of a crew.

        This is only collected if the user has opted-in to share the crew.

        Args:
            crew: The crew being executed.
            inputs: Optional input parameters for the crew.

        Returns:
            The execution span if crew sharing is enabled, None otherwise.
        """
        self.crew_creation(crew, inputs)

        def _operation() -> Span:
            tracer = trace.get_tracer("crewai.telemetry")
            span = tracer.start_span("Crew Execution")
            self._add_attribute(
                span,
                "crewai_version",
                version("crewai"),
            )
            add_crew_attributes(
                span, crew, self._add_attribute, include_fingerprint=False
            )
            self._add_attribute(span, "crew_inputs", json.dumps(inputs or {}))
            self._add_attribute(
                span,
                "crew_agents",
                json.dumps(
                    [
                        {
                            "key": agent.key,
                            "id": str(agent.id),
                            "role": agent.role,
                            "goal": agent.goal,
                            "backstory": agent.backstory,
                            "verbose?": agent.verbose,
                            "max_iter": agent.max_iter,
                            "max_rpm": agent.max_rpm,
                            "i18n": agent.i18n.prompt_file,
                            "llm": agent.llm.model,
                            "delegation_enabled?": agent.allow_delegation,
                            "tools_names": [
                                sanitize_tool_name(tool.name)
                                for tool in agent.tools or []
                            ],
                        }
                        for agent in crew.agents
                    ]
                ),
            )
            self._add_attribute(
                span,
                "crew_tasks",
                json.dumps(
                    [
                        {
                            "id": str(task.id),
                            "description": task.description,
                            "expected_output": task.expected_output,
                            "async_execution?": task.async_execution,
                            "human_input?": task.human_input,
                            "agent_role": task.agent.role if task.agent else "None",
                            "agent_key": task.agent.key if task.agent else None,
                            "context": (
                                [task.description for task in task.context]
                                if isinstance(task.context, list)
                                else None
                            ),
                            "tools_names": [
                                sanitize_tool_name(tool.name)
                                for tool in task.tools or []
                            ],
                        }
                        for task in crew.tasks
                    ]
                ),
            )
            return span

        if crew.share_crew:
            return self._safe_telemetry_operation(_operation)
        return None

    def end_crew(self, crew: Any, final_string_output: str) -> None:
        """Records the end of crew execution.

        Args:
            crew: The crew that finished execution.
            final_string_output: The final output from the crew.
        """

        def _operation() -> None:
            self._add_attribute(
                crew._execution_span,
                "crewai_version",
                version("crewai"),
            )
            self._add_attribute(
                crew._execution_span, "crew_output", final_string_output
            )
            self._add_attribute(
                crew._execution_span,
                "crew_tasks_output",
                json.dumps(
                    [
                        {
                            "id": str(task.id),
                            "description": task.description,
                            "output": task.output.raw_output,
                        }
                        for task in crew.tasks
                    ]
                ),
            )
            close_span(crew._execution_span)

        if crew.share_crew:
            self._safe_telemetry_operation(_operation)

    def _add_attribute(self, span: Span, key: str, value: Any) -> None:
        """Add an attribute to a span.

        Args:
            span: The span to add the attribute to.
            key: The attribute key.
            value: The attribute value.
        """

        def _operation() -> None:
            return span.set_attribute(key, value)

        self._safe_telemetry_operation(_operation)

    def flow_creation_span(self, flow_name: str) -> None:
        """Records the creation of a new flow.

        Args:
            flow_name: Name of the flow being created.
        """

        def _operation() -> None:
            tracer = trace.get_tracer("crewai.telemetry")
            span = tracer.start_span("Flow Creation")
            self._add_attribute(span, "flow_name", flow_name)
            close_span(span)

        self._safe_telemetry_operation(_operation)

    def flow_plotting_span(self, flow_name: str, node_names: list[str]) -> None:
        """Records flow visualization/plotting activity.

        Args:
            flow_name: Name of the flow being plotted.
            node_names: List of node names in the flow.
        """

        def _operation() -> None:
            tracer = trace.get_tracer("crewai.telemetry")
            span = tracer.start_span("Flow Plotting")
            self._add_attribute(span, "flow_name", flow_name)
            self._add_attribute(span, "node_names", json.dumps(node_names))
            close_span(span)

        self._safe_telemetry_operation(_operation)

    def flow_execution_span(self, flow_name: str, node_names: list[str]) -> None:
        """Records the execution of a flow.

        Args:
            flow_name: Name of the flow being executed.
            node_names: List of nodes being executed in the flow.
        """

        def _operation() -> None:
            tracer = trace.get_tracer("crewai.telemetry")
            span = tracer.start_span("Flow Execution")
            self._add_attribute(span, "flow_name", flow_name)
            self._add_attribute(span, "node_names", json.dumps(node_names))
            close_span(span)

        self._safe_telemetry_operation(_operation)

    def human_feedback_span(
        self,
        event_type: str,
        has_routing: bool,
        num_outcomes: int = 0,
        feedback_provided: bool | None = None,
        outcome: str | None = None,
    ) -> None:
        """Records human feedback feature usage.

        Args:
            event_type: Type of event - "requested" or "received".
            has_routing: Whether emit options were configured for routing.
            num_outcomes: Number of possible outcomes if routing is used.
            feedback_provided: Whether user provided feedback or skipped (None if requested).
            outcome: The collapsed outcome string if routing was used.
        """

        def _operation() -> None:
            tracer = trace.get_tracer("crewai.telemetry")
            span = tracer.start_span("Human Feedback")
            self._add_attribute(span, "event_type", event_type)
            self._add_attribute(span, "has_routing", has_routing)
            self._add_attribute(span, "num_outcomes", num_outcomes)
            if feedback_provided is not None:
                self._add_attribute(span, "feedback_provided", feedback_provided)
            if outcome is not None:
                self._add_attribute(span, "outcome", outcome)
            close_span(span)

        self._safe_telemetry_operation(_operation)
