from __future__ import annotations

import asyncio
import json
import os
import platform
import warnings
from contextlib import contextmanager
from importlib.metadata import version
from typing import TYPE_CHECKING, Any, Optional

from crewai.telemetry.constants import (
    CREWAI_TELEMETRY_BASE_URL,
    CREWAI_TELEMETRY_SERVICE_NAME,
)


@contextmanager
def suppress_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        yield


from opentelemetry import trace  # noqa: E402
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,  # noqa: E402
)
from opentelemetry.sdk.resources import SERVICE_NAME, Resource  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import BatchSpanProcessor  # noqa: E402
from opentelemetry.trace import Span, Status, StatusCode  # noqa: E402

if TYPE_CHECKING:
    from crewai.crew import Crew
    from crewai.task import Task


class Telemetry:
    """A class to handle anonymous telemetry for the crewai package.

    The data being collected is for development purpose, all data is anonymous.

    There is NO data being collected on the prompts, tasks descriptions
    agents backstories or goals nor responses or any data that is being
    processed by the agents, nor any secrets and env vars.

    Users can opt-in to sharing more complete data using the `share_crew`
    attribute in the Crew class.
    """

    def __init__(self):
        self.ready: bool = False
        self.trace_set: bool = False

        if self._is_telemetry_disabled():
            return

        try:
            self.resource = Resource(
                attributes={SERVICE_NAME: CREWAI_TELEMETRY_SERVICE_NAME},
            )
            with suppress_warnings():
                self.provider = TracerProvider(resource=self.resource)

            processor = BatchSpanProcessor(
                OTLPSpanExporter(
                    endpoint=f"{CREWAI_TELEMETRY_BASE_URL}/v1/traces",
                    timeout=30,
                )
            )

            self.provider.add_span_processor(processor)
            self.ready = True
        except Exception as e:
            if isinstance(
                e,
                (SystemExit, KeyboardInterrupt, GeneratorExit, asyncio.CancelledError),
            ):
                raise  # Re-raise the exception to not interfere with system signals
            self.ready = False

    def _is_telemetry_disabled(self) -> bool:
        """Check if telemetry should be disabled based on environment variables."""
        return (
            os.getenv("OTEL_SDK_DISABLED", "false").lower() == "true"
            or os.getenv("CREWAI_DISABLE_TELEMETRY", "false").lower() == "true"
        )

    def set_tracer(self):
        if self.ready and not self.trace_set:
            try:
                with suppress_warnings():
                    trace.set_tracer_provider(self.provider)
                    self.trace_set = True
            except Exception:
                self.ready = False
                self.trace_set = False

    def _safe_telemetry_operation(self, operation):
        if not self.ready:
            return
        try:
            operation()
        except Exception:
            pass

    def crew_creation(self, crew: Crew, inputs: dict[str, Any] | None):
        """Records the creation of a crew."""

        def operation():
            tracer = trace.get_tracer("crewai.telemetry")
            span = tracer.start_span("Crew Created")
            self._add_attribute(
                span,
                "crewai_version",
                version("crewai"),
            )
            self._add_attribute(span, "python_version", platform.python_version())
            self._add_attribute(span, "crew_key", crew.key)
            self._add_attribute(span, "crew_id", str(crew.id))
            self._add_attribute(span, "crew_process", crew.process)
            self._add_attribute(span, "crew_memory", crew.memory)
            self._add_attribute(span, "crew_number_of_tasks", len(crew.tasks))
            self._add_attribute(span, "crew_number_of_agents", len(crew.agents))

            # Add fingerprint data
            if hasattr(crew, "fingerprint") and crew.fingerprint:
                self._add_attribute(span, "crew_fingerprint", crew.fingerprint.uuid_str)
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
                                    tool.name.casefold() for tool in agent.tools or []
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
                                    if task.context
                                    else None
                                ),
                                "tools_names": [
                                    tool.name.casefold() for tool in task.tools or []
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
                self._add_attribute(
                    span, "crew_inputs", json.dumps(inputs) if inputs else None
                )
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
                                    tool.name.casefold() for tool in agent.tools or []
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
                                    tool.name.casefold() for tool in task.tools or []
                                ],
                            }
                            for task in crew.tasks
                        ]
                    ),
                )
            span.set_status(Status(StatusCode.OK))
            span.end()

        self._safe_telemetry_operation(operation)

    def task_started(self, crew: Crew, task: Task) -> Span | None:
        """Records task started in a crew."""

        def operation():
            tracer = trace.get_tracer("crewai.telemetry")

            created_span = tracer.start_span("Task Created")

            self._add_attribute(created_span, "crew_key", crew.key)
            self._add_attribute(created_span, "crew_id", str(crew.id))
            self._add_attribute(created_span, "task_key", task.key)
            self._add_attribute(created_span, "task_id", str(task.id))

            # Add fingerprint data
            if hasattr(crew, "fingerprint") and crew.fingerprint:
                self._add_attribute(
                    created_span, "crew_fingerprint", crew.fingerprint.uuid_str
                )

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
                agent_fingerprint = getattr(
                    getattr(task.agent, "fingerprint", None), "uuid_str", None
                )
                if agent_fingerprint:
                    self._add_attribute(
                        created_span, "agent_fingerprint", agent_fingerprint
                    )

            if crew.share_crew:
                self._add_attribute(
                    created_span, "formatted_description", task.description
                )
                self._add_attribute(
                    created_span, "formatted_expected_output", task.expected_output
                )

            created_span.set_status(Status(StatusCode.OK))
            created_span.end()

            span = tracer.start_span("Task Execution")

            self._add_attribute(span, "crew_key", crew.key)
            self._add_attribute(span, "crew_id", str(crew.id))
            self._add_attribute(span, "task_key", task.key)
            self._add_attribute(span, "task_id", str(task.id))

            # Add fingerprint data to execution span
            if hasattr(crew, "fingerprint") and crew.fingerprint:
                self._add_attribute(span, "crew_fingerprint", crew.fingerprint.uuid_str)

            if hasattr(task, "fingerprint") and task.fingerprint:
                self._add_attribute(span, "task_fingerprint", task.fingerprint.uuid_str)

            # Add agent fingerprint if task has an assigned agent
            if hasattr(task, "agent") and task.agent:
                agent_fingerprint = getattr(
                    getattr(task.agent, "fingerprint", None), "uuid_str", None
                )
                if agent_fingerprint:
                    self._add_attribute(span, "agent_fingerprint", agent_fingerprint)

            if crew.share_crew:
                self._add_attribute(span, "formatted_description", task.description)
                self._add_attribute(
                    span, "formatted_expected_output", task.expected_output
                )

            return span

        return self._safe_telemetry_operation(operation)

    def task_ended(self, span: Span, task: Task, crew: Crew):
        """Records the completion of a task execution in a crew.

        Args:
            span (Span): The OpenTelemetry span tracking the task execution
            task (Task): The task that was completed
            crew (Crew): The crew context in which the task was executed

        Note:
            If share_crew is enabled, this will also record the task output
        """

        def operation():
            # Ensure fingerprint data is present on completion span
            if hasattr(task, "fingerprint") and task.fingerprint:
                self._add_attribute(span, "task_fingerprint", task.fingerprint.uuid_str)

            if crew.share_crew:
                self._add_attribute(
                    span,
                    "task_output",
                    task.output.raw if task.output else "",
                )

            span.set_status(Status(StatusCode.OK))
            span.end()

        self._safe_telemetry_operation(operation)

    def tool_repeated_usage(self, llm: Any, tool_name: str, attempts: int):
        """Records when a tool is used repeatedly, which might indicate an issue.

        Args:
            llm (Any): The language model being used
            tool_name (str): Name of the tool being repeatedly used
            attempts (int): Number of attempts made with this tool
        """

        def operation():
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
            span.set_status(Status(StatusCode.OK))
            span.end()

        self._safe_telemetry_operation(operation)

    def tool_usage(self, llm: Any, tool_name: str, attempts: int, agent: Any = None):
        """Records the usage of a tool by an agent.

        Args:
            llm (Any): The language model being used
            tool_name (str): Name of the tool being used
            attempts (int): Number of attempts made with this tool
            agent (Any, optional): The agent using the tool
        """

        def operation():
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
            if agent and hasattr(agent, "fingerprint") and agent.fingerprint:
                self._add_attribute(
                    span, "agent_fingerprint", agent.fingerprint.uuid_str
                )
                if hasattr(agent, "role"):
                    self._add_attribute(span, "agent_role", agent.role)

            span.set_status(Status(StatusCode.OK))
            span.end()

        self._safe_telemetry_operation(operation)

    def tool_usage_error(
        self, llm: Any, agent: Any = None, tool_name: Optional[str] = None
    ):
        """Records when a tool usage results in an error.

        Args:
            llm (Any): The language model being used when the error occurred
            agent (Any, optional): The agent using the tool
            tool_name (str, optional): Name of the tool that caused the error
        """

        def operation():
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
            if agent and hasattr(agent, "fingerprint") and agent.fingerprint:
                self._add_attribute(
                    span, "agent_fingerprint", agent.fingerprint.uuid_str
                )
                if hasattr(agent, "role"):
                    self._add_attribute(span, "agent_role", agent.role)

            span.set_status(Status(StatusCode.OK))
            span.end()

        self._safe_telemetry_operation(operation)

    def individual_test_result_span(
        self, crew: Crew, quality: float, exec_time: int, model_name: str
    ):
        """Records individual test results for a crew execution.

        Args:
            crew (Crew): The crew being tested
            quality (float): Quality score of the execution
            exec_time (int): Execution time in seconds
            model_name (str): Name of the model used
        """

        def operation():
            tracer = trace.get_tracer("crewai.telemetry")
            span = tracer.start_span("Crew Individual Test Result")

            self._add_attribute(
                span,
                "crewai_version",
                version("crewai"),
            )
            self._add_attribute(span, "crew_key", crew.key)
            self._add_attribute(span, "crew_id", str(crew.id))
            self._add_attribute(span, "quality", str(quality))
            self._add_attribute(span, "exec_time", str(exec_time))
            self._add_attribute(span, "model_name", model_name)
            span.set_status(Status(StatusCode.OK))
            span.end()

        self._safe_telemetry_operation(operation)

    def test_execution_span(
        self,
        crew: Crew,
        iterations: int,
        inputs: dict[str, Any] | None,
        model_name: str,
    ):
        """Records the execution of a test suite for a crew.

        Args:
            crew (Crew): The crew being tested
            iterations (int): Number of test iterations
            inputs (dict[str, Any] | None): Input parameters for the test
            model_name (str): Name of the model used in testing
        """

        def operation():
            tracer = trace.get_tracer("crewai.telemetry")
            span = tracer.start_span("Crew Test Execution")

            self._add_attribute(
                span,
                "crewai_version",
                version("crewai"),
            )
            self._add_attribute(span, "crew_key", crew.key)
            self._add_attribute(span, "crew_id", str(crew.id))
            self._add_attribute(span, "iterations", str(iterations))
            self._add_attribute(span, "model_name", model_name)

            if crew.share_crew:
                self._add_attribute(
                    span, "inputs", json.dumps(inputs) if inputs else None
                )

            span.set_status(Status(StatusCode.OK))
            span.end()

        self._safe_telemetry_operation(operation)

    def deploy_signup_error_span(self):
        """Records when an error occurs during the deployment signup process."""

        def operation():
            tracer = trace.get_tracer("crewai.telemetry")
            span = tracer.start_span("Deploy Signup Error")
            span.set_status(Status(StatusCode.OK))
            span.end()

        self._safe_telemetry_operation(operation)

    def start_deployment_span(self, uuid: Optional[str] = None):
        """Records the start of a deployment process.

        Args:
            uuid (Optional[str]): Unique identifier for the deployment
        """

        def operation():
            tracer = trace.get_tracer("crewai.telemetry")
            span = tracer.start_span("Start Deployment")
            if uuid:
                self._add_attribute(span, "uuid", uuid)
            span.set_status(Status(StatusCode.OK))
            span.end()

        self._safe_telemetry_operation(operation)

    def create_crew_deployment_span(self):
        """Records the creation of a new crew deployment."""

        def operation():
            tracer = trace.get_tracer("crewai.telemetry")
            span = tracer.start_span("Create Crew Deployment")
            span.set_status(Status(StatusCode.OK))
            span.end()

        self._safe_telemetry_operation(operation)

    def get_crew_logs_span(self, uuid: Optional[str], log_type: str = "deployment"):
        """Records the retrieval of crew logs.

        Args:
            uuid (Optional[str]): Unique identifier for the crew
            log_type (str, optional): Type of logs being retrieved. Defaults to "deployment".
        """

        def operation():
            tracer = trace.get_tracer("crewai.telemetry")
            span = tracer.start_span("Get Crew Logs")
            self._add_attribute(span, "log_type", log_type)
            if uuid:
                self._add_attribute(span, "uuid", uuid)
            span.set_status(Status(StatusCode.OK))
            span.end()

        self._safe_telemetry_operation(operation)

    def remove_crew_span(self, uuid: Optional[str] = None):
        """Records the removal of a crew.

        Args:
            uuid (Optional[str]): Unique identifier for the crew being removed
        """

        def operation():
            tracer = trace.get_tracer("crewai.telemetry")
            span = tracer.start_span("Remove Crew")
            if uuid:
                self._add_attribute(span, "uuid", uuid)
            span.set_status(Status(StatusCode.OK))
            span.end()

        self._safe_telemetry_operation(operation)

    def crew_execution_span(self, crew: Crew, inputs: dict[str, Any] | None):
        """Records the complete execution of a crew.
        This is only collected if the user has opted-in to share the crew.
        """
        self.crew_creation(crew, inputs)

        def operation():
            tracer = trace.get_tracer("crewai.telemetry")
            span = tracer.start_span("Crew Execution")
            self._add_attribute(
                span,
                "crewai_version",
                version("crewai"),
            )
            self._add_attribute(span, "crew_key", crew.key)
            self._add_attribute(span, "crew_id", str(crew.id))
            self._add_attribute(
                span, "crew_inputs", json.dumps(inputs) if inputs else None
            )
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
                                tool.name.casefold() for tool in agent.tools or []
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
                                if task.context
                                else None
                            ),
                            "tools_names": [
                                tool.name.casefold() for tool in task.tools or []
                            ],
                        }
                        for task in crew.tasks
                    ]
                ),
            )
            return span

        if crew.share_crew:
            return self._safe_telemetry_operation(operation)
        return None

    def end_crew(self, crew, final_string_output):
        def operation():
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
            crew._execution_span.set_status(Status(StatusCode.OK))
            crew._execution_span.end()

        if crew.share_crew:
            self._safe_telemetry_operation(operation)

    def _add_attribute(self, span, key, value):
        """Add an attribute to a span."""

        def operation():
            return span.set_attribute(key, value)

        self._safe_telemetry_operation(operation)

    def flow_creation_span(self, flow_name: str):
        """Records the creation of a new flow.

        Args:
            flow_name (str): Name of the flow being created
        """

        def operation():
            tracer = trace.get_tracer("crewai.telemetry")
            span = tracer.start_span("Flow Creation")
            self._add_attribute(span, "flow_name", flow_name)
            span.set_status(Status(StatusCode.OK))
            span.end()

        self._safe_telemetry_operation(operation)

    def flow_plotting_span(self, flow_name: str, node_names: list[str]):
        """Records flow visualization/plotting activity.

        Args:
            flow_name (str): Name of the flow being plotted
            node_names (list[str]): List of node names in the flow
        """

        def operation():
            tracer = trace.get_tracer("crewai.telemetry")
            span = tracer.start_span("Flow Plotting")
            self._add_attribute(span, "flow_name", flow_name)
            self._add_attribute(span, "node_names", json.dumps(node_names))
            span.set_status(Status(StatusCode.OK))
            span.end()

        self._safe_telemetry_operation(operation)

    def flow_execution_span(self, flow_name: str, node_names: list[str]):
        """Records the execution of a flow.

        Args:
            flow_name (str): Name of the flow being executed
            node_names (list[str]): List of nodes being executed in the flow
        """

        def operation():
            tracer = trace.get_tracer("crewai.telemetry")
            span = tracer.start_span("Flow Execution")
            self._add_attribute(span, "flow_name", flow_name)
            self._add_attribute(span, "node_names", json.dumps(node_names))
            span.set_status(Status(StatusCode.OK))
            span.end()

        self._safe_telemetry_operation(operation)
