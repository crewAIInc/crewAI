from __future__ import annotations

from concurrent.futures import Future
from copy import copy as shallow_copy
import datetime
from hashlib import md5
import inspect
import json
import logging
from pathlib import Path
import threading
from typing import (
    Any,
    ClassVar,
    cast,
    get_args,
    get_origin,
)
import uuid
import warnings

from pydantic import (
    UUID4,
    BaseModel,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError
from typing_extensions import Self

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.task_events import (
    TaskCompletedEvent,
    TaskFailedEvent,
    TaskStartedEvent,
)
from crewai.security import Fingerprint, SecurityConfig
from crewai.tasks.output_format import OutputFormat
from crewai.tasks.task_output import TaskOutput
from crewai.tools.base_tool import BaseTool
from crewai.utilities.config import process_config
from crewai.utilities.constants import NOT_SPECIFIED, _NotSpecified
from crewai.utilities.converter import Converter, convert_to_model
from crewai.utilities.file_store import (
    clear_task_files,
    get_all_files,
    store_task_files,
)


try:
    from crewai_files import FileInput, FilePath, get_supported_content_types

    HAS_CREWAI_FILES = True
except ImportError:
    FileInput = Any  # type: ignore[misc,assignment]
    HAS_CREWAI_FILES = False

    def get_supported_content_types(provider: str, api: str | None = None) -> list[str]:
        return []


from crewai.utilities.guardrail import (
    process_guardrail,
)
from crewai.utilities.guardrail_types import (
    GuardrailCallable,
    GuardrailType,
    GuardrailsType,
)
from crewai.utilities.i18n import I18N, get_i18n
from crewai.utilities.printer import Printer
from crewai.utilities.string_utils import interpolate_only


_printer = Printer()


class Task(BaseModel):
    """Class that represents a task to be executed.

    Each task must have a description, an expected output and an agent responsible for execution.

    Attributes:
        agent: Agent responsible for task execution. Represents entity performing task.
        async_execution: Boolean flag indicating asynchronous task execution.
        callback: Function/object executed post task completion for additional actions.
        config: Dictionary containing task-specific configuration parameters.
        context: List of Task instances providing task context or input data.
        description: Descriptive text detailing task's purpose and execution.
        expected_output: Clear definition of expected task outcome.
        output_file: File path for storing task output.
        create_directory: Whether to create the directory for output_file if it doesn't exist.
        output_json: Pydantic model for structuring JSON output.
        output_pydantic: Pydantic model for task output.
        security_config: Security configuration including fingerprinting.
        tools: List of tools/resources limited for task execution.
        allow_crewai_trigger_context: Optional flag to control crewai_trigger_payload injection.
                              None (default): Auto-inject for first task only.
                              True: Always inject trigger payload for this task.
                              False: Never inject trigger payload, even for first task.
    """

    __hash__ = object.__hash__
    logger: ClassVar[logging.Logger] = logging.getLogger(__name__)
    used_tools: int = 0
    tools_errors: int = 0
    delegations: int = 0
    i18n: I18N = Field(default_factory=get_i18n)
    name: str | None = Field(default=None)
    prompt_context: str | None = None
    description: str = Field(description="Description of the actual task.")
    expected_output: str = Field(
        description="Clear definition of expected output for the task."
    )
    config: dict[str, Any] | None = Field(
        description="Configuration for the agent",
        default=None,
    )
    callback: Any | None = Field(
        description="Callback to be executed after the task is completed.", default=None
    )
    agent: BaseAgent | None = Field(
        description="Agent responsible for execution the task.", default=None
    )
    context: list[Task] | None | _NotSpecified = Field(
        description="Other tasks that will have their output used as context for this task.",
        default=NOT_SPECIFIED,
    )
    async_execution: bool | None = Field(
        description="Whether the task should be executed asynchronously or not.",
        default=False,
    )
    output_json: type[BaseModel] | None = Field(
        description="A Pydantic model to be used to create a JSON output.",
        default=None,
    )
    output_pydantic: type[BaseModel] | None = Field(
        description="A Pydantic model to be used to create a Pydantic output.",
        default=None,
    )
    response_model: type[BaseModel] | None = Field(
        description="A Pydantic model for structured LLM outputs using native provider features.",
        default=None,
    )
    output_file: str | None = Field(
        description="A file path to be used to create a file output.",
        default=None,
    )
    create_directory: bool | None = Field(
        description="Whether to create the directory for output_file if it doesn't exist.",
        default=True,
    )
    output: TaskOutput | None = Field(
        description="Task output, it's final result after being executed", default=None
    )
    tools: list[BaseTool] | None = Field(
        default_factory=list,
        description="Tools the agent is limited to use for this task.",
    )
    input_files: dict[str, FileInput] = Field(
        default_factory=dict,
        description="Named input files for this task. Keys are reference names, values are paths or File objects.",
    )
    security_config: SecurityConfig = Field(
        default_factory=SecurityConfig,
        description="Security configuration for the task.",
    )
    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        frozen=True,
        description="Unique identifier for the object, not set by user.",
    )
    human_input: bool | None = Field(
        description="Whether the task should have a human review the final answer of the agent",
        default=False,
    )
    markdown: bool | None = Field(
        description="Whether the task should instruct the agent to return the final answer formatted in Markdown",
        default=False,
    )
    converter_cls: type[Converter] | None = Field(
        description="A converter class used to export structured output",
        default=None,
    )
    processed_by_agents: set[str] = Field(default_factory=set)
    guardrail: GuardrailType | None = Field(
        default=None,
        description="Function or string description of a guardrail to validate task output before proceeding to next task",
    )
    guardrails: GuardrailsType | None = Field(
        default=None,
        description="List of guardrails to validate task output before proceeding to next task. Also supports a single guardrail function or string description of a guardrail to validate task output before proceeding to next task",
    )

    max_retries: int | None = Field(
        default=None,
        description="[DEPRECATED] Maximum number of retries when guardrail fails. Use guardrail_max_retries instead. Will be removed in v1.0.0",
    )
    guardrail_max_retries: int = Field(
        default=3, description="Maximum number of retries when guardrail fails"
    )
    retry_count: int = Field(default=0, description="Current number of retries")
    start_time: datetime.datetime | None = Field(
        default=None, description="Start time of the task execution"
    )
    end_time: datetime.datetime | None = Field(
        default=None, description="End time of the task execution"
    )
    allow_crewai_trigger_context: bool | None = Field(
        default=None,
        description="Whether this task should append 'Trigger Payload: {crewai_trigger_payload}' to the task description when crewai_trigger_payload exists in crew inputs.",
    )
    _guardrail: GuardrailCallable | None = PrivateAttr(default=None)
    _guardrails: list[GuardrailCallable] = PrivateAttr(
        default_factory=list,
    )
    _guardrail_retry_counts: dict[int, int] = PrivateAttr(
        default_factory=dict,
    )
    _original_description: str | None = PrivateAttr(default=None)
    _original_expected_output: str | None = PrivateAttr(default=None)
    _original_output_file: str | None = PrivateAttr(default=None)
    _thread: threading.Thread | None = PrivateAttr(default=None)
    model_config = {"arbitrary_types_allowed": True}

    @field_validator("guardrail")
    @classmethod
    def validate_guardrail_function(
        cls, v: str | GuardrailCallable | None
    ) -> str | GuardrailCallable | None:
        """
        If v is a callable, validate that the guardrail function has the correct signature and behavior.
        If v is a string, return it as is.

        While type hints provide static checking, this validator ensures runtime safety by:
        1. Verifying the function accepts exactly one parameter (the TaskOutput)
        2. Checking return type annotations match Tuple[bool, Any] if present
        3. Providing clear, immediate error messages for debugging

        This runtime validation is crucial because:
        - Type hints are optional and can be ignored at runtime
        - Function signatures need immediate validation before task execution
        - Clear error messages help users debug guardrail implementation issues

        Args:
            v: The guardrail function to validate or a string describing the guardrail task

        Returns:
            The validated guardrail function or a string describing the guardrail task

        Raises:
            ValueError: If the function signature is invalid or return annotation
                       doesn't match Tuple[bool, Any]
        """
        if v is not None and callable(v):
            sig = inspect.signature(v)
            positional_args = [
                param
                for param in sig.parameters.values()
                if param.default is inspect.Parameter.empty
            ]
            if len(positional_args) != 1:
                raise ValueError("Guardrail function must accept exactly one parameter")

            # Check return annotation if present, but don't require it
            return_annotation = sig.return_annotation
            if return_annotation != inspect.Signature.empty:
                return_annotation_args = get_args(return_annotation)
                if not (
                    get_origin(return_annotation) is tuple
                    and len(return_annotation_args) == 2
                    and return_annotation_args[0] is bool
                    and (
                        return_annotation_args[1] is Any
                        or return_annotation_args[1] is str
                        or return_annotation_args[1] is TaskOutput
                        or return_annotation_args[1] == str | TaskOutput
                    )
                ):
                    raise ValueError(
                        "If return type is annotated, it must be Tuple[bool, Any]"
                    )
        return v

    @model_validator(mode="before")
    @classmethod
    def process_model_config(cls, values: dict[str, Any]) -> dict[str, Any]:
        return process_config(values, cls)

    @model_validator(mode="after")
    def validate_required_fields(self) -> Self:
        required_fields = ["description", "expected_output"]
        for field in required_fields:
            if getattr(self, field) is None:
                raise ValueError(
                    f"{field} must be provided either directly or through config"
                )
        return self

    @model_validator(mode="after")
    def ensure_guardrail_is_callable(self) -> Task:
        if callable(self.guardrail):
            self._guardrail = self.guardrail
        elif isinstance(self.guardrail, str):
            from crewai.tasks.llm_guardrail import LLMGuardrail

            if self.agent is None:
                raise ValueError("Agent is required to use LLMGuardrail")

            self._guardrail = cast(
                GuardrailCallable,
                LLMGuardrail(description=self.guardrail, llm=self.agent.llm),
            )

        return self

    @model_validator(mode="after")
    def ensure_guardrails_is_list_of_callables(self) -> Task:
        guardrails = []
        if self.guardrails is not None:
            if isinstance(self.guardrails, (list, tuple)):
                if len(self.guardrails) > 0:
                    for guardrail in self.guardrails:
                        if callable(guardrail):
                            guardrails.append(guardrail)
                        elif isinstance(guardrail, str):
                            if self.agent is None:
                                raise ValueError(
                                    "Agent is required to use non-programmatic guardrails"
                                )
                            from crewai.tasks.llm_guardrail import LLMGuardrail

                            guardrails.append(
                                cast(
                                    GuardrailCallable,
                                    LLMGuardrail(
                                        description=guardrail, llm=self.agent.llm
                                    ),
                                )
                            )
                        else:
                            raise ValueError("Guardrail must be a callable or a string")
            else:
                if callable(self.guardrails):
                    guardrails.append(self.guardrails)
                elif isinstance(self.guardrails, str):
                    if self.agent is None:
                        raise ValueError(
                            "Agent is required to use non-programmatic guardrails"
                        )
                    from crewai.tasks.llm_guardrail import LLMGuardrail

                    guardrails.append(
                        cast(
                            GuardrailCallable,
                            LLMGuardrail(
                                description=self.guardrails, llm=self.agent.llm
                            ),
                        )
                    )
                else:
                    raise ValueError("Guardrail must be a callable or a string")

        self._guardrails = guardrails
        if self._guardrails:
            self.guardrail = None
            self._guardrail = None

        return self

    @field_validator("id", mode="before")
    @classmethod
    def _deny_user_set_id(cls, v: UUID4 | None) -> None:
        if v:
            raise PydanticCustomError(
                "may_not_set_field", "This field is not to be set by the user.", {}
            )

    @field_validator("input_files", mode="before")
    @classmethod
    def _normalize_input_files(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Convert string paths to FilePath objects."""
        if not v:
            return v

        if not HAS_CREWAI_FILES:
            return v

        result = {}
        for key, value in v.items():
            if isinstance(value, str):
                result[key] = FilePath(path=Path(value))
            else:
                result[key] = value
        return result

    @field_validator("output_file")
    @classmethod
    def output_file_validation(cls, value: str | None) -> str | None:
        """Validate the output file path.

        Args:
            value: The output file path to validate. Can be None or a string.
                  If the path contains template variables (e.g. {var}), leading slashes are preserved.
                  For regular paths, leading slashes are stripped.

        Returns:
            The validated and potentially modified path, or None if no path was provided.

        Raises:
            ValueError: If the path contains invalid characters, path traversal attempts,
                      or other security concerns.
        """
        if value is None:
            return None

        # Basic security checks
        if ".." in value:
            raise ValueError(
                "Path traversal attempts are not allowed in output_file paths"
            )

        # Check for shell expansion first
        if value.startswith(("~", "$")):
            raise ValueError(
                "Shell expansion characters are not allowed in output_file paths"
            )

        # Then check other shell special characters
        if any(char in value for char in ["|", ">", "<", "&", ";"]):
            raise ValueError(
                "Shell special characters are not allowed in output_file paths"
            )

        # Don't strip leading slash if it's a template path with variables
        if "{" in value or "}" in value:
            # Validate template variable format
            template_vars = [part.split("}")[0] for part in value.split("{")[1:]]
            for var in template_vars:
                if not var.isidentifier():
                    raise ValueError(f"Invalid template variable name: {var}")
            return value

        # Strip leading slash for regular paths
        if value.startswith("/"):
            return value[1:]
        return value

    @model_validator(mode="after")
    def set_attributes_based_on_config(self) -> Task:
        """Set attributes based on the agent configuration."""
        if self.config:
            for key, value in self.config.items():
                setattr(self, key, value)
        return self

    @model_validator(mode="after")
    def check_tools(self) -> Self:
        """Check if the tools are set."""
        if not self.tools and self.agent and self.agent.tools:
            self.tools = self.agent.tools
        return self

    @model_validator(mode="after")
    def check_output(self) -> Self:
        """Check if an output type is set."""
        output_types = [self.output_json, self.output_pydantic]
        if len([type for type in output_types if type]) > 1:
            raise PydanticCustomError(
                "output_type",
                "Only one output type can be set, either output_pydantic or output_json.",
                {},
            )
        return self

    @model_validator(mode="after")
    def handle_max_retries_deprecation(self) -> Self:
        if self.max_retries is not None:
            warnings.warn(
                "The 'max_retries' parameter is deprecated and will be removed in CrewAI v1.0.0. "
                "Please use 'guardrail_max_retries' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.guardrail_max_retries = self.max_retries
        return self

    def execute_sync(
        self,
        agent: BaseAgent | None = None,
        context: str | None = None,
        tools: list[BaseTool] | None = None,
    ) -> TaskOutput:
        """Execute the task synchronously."""
        return self._execute_core(agent, context, tools)

    @property
    def key(self) -> str:
        description = self._original_description or self.description
        expected_output = self._original_expected_output or self.expected_output
        source = [description, expected_output]

        return md5("|".join(source).encode(), usedforsecurity=False).hexdigest()

    @property
    def execution_duration(self) -> float | None:
        if not self.start_time or not self.end_time:
            return None
        return (self.end_time - self.start_time).total_seconds()

    def execute_async(
        self,
        agent: BaseAgent | None = None,
        context: str | None = None,
        tools: list[BaseTool] | None = None,
    ) -> Future[TaskOutput]:
        """Execute the task asynchronously."""
        future: Future[TaskOutput] = Future()
        threading.Thread(
            daemon=True,
            target=self._execute_task_async,
            args=(agent, context, tools, future),
        ).start()
        return future

    def _execute_task_async(
        self,
        agent: BaseAgent | None,
        context: str | None,
        tools: list[Any] | None,
        future: Future[TaskOutput],
    ) -> None:
        """Execute the task asynchronously with context handling."""
        try:
            result = self._execute_core(agent, context, tools)
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)

    async def aexecute_sync(
        self,
        agent: BaseAgent | None = None,
        context: str | None = None,
        tools: list[BaseTool] | None = None,
    ) -> TaskOutput:
        """Execute the task asynchronously using native async/await."""
        return await self._aexecute_core(agent, context, tools)

    async def _aexecute_core(
        self,
        agent: BaseAgent | None,
        context: str | None,
        tools: list[Any] | None,
    ) -> TaskOutput:
        """Run the core execution logic of the task asynchronously."""
        self._store_input_files()
        try:
            agent = agent or self.agent
            self.agent = agent
            if not agent:
                raise Exception(
                    f"The task '{self.description}' has no agent assigned, therefore it can't be executed directly and should be executed in a Crew using a specific process that support that, like hierarchical."
                )

            self.start_time = datetime.datetime.now()

            self.prompt_context = context
            tools = tools or self.tools or []

            self.processed_by_agents.add(agent.role)
            crewai_event_bus.emit(self, TaskStartedEvent(context=context, task=self))  # type: ignore[no-untyped-call]
            result = await agent.aexecute_task(
                task=self,
                context=context,
                tools=tools,
            )

            if not self._guardrails and not self._guardrail:
                pydantic_output, json_output = self._export_output(result)
            else:
                pydantic_output, json_output = None, None

            task_output = TaskOutput(
                name=self.name or self.description,
                description=self.description,
                expected_output=self.expected_output,
                raw=result,
                pydantic=pydantic_output,
                json_dict=json_output,
                agent=agent.role,
                output_format=self._get_output_format(),
                messages=agent.last_messages,  # type: ignore[attr-defined]
            )

            if self._guardrails:
                for idx, guardrail in enumerate(self._guardrails):
                    task_output = await self._ainvoke_guardrail_function(
                        task_output=task_output,
                        agent=agent,
                        tools=tools,
                        guardrail=guardrail,
                        guardrail_index=idx,
                    )

            if self._guardrail:
                task_output = await self._ainvoke_guardrail_function(
                    task_output=task_output,
                    agent=agent,
                    tools=tools,
                    guardrail=self._guardrail,
                )

            self.output = task_output
            self.end_time = datetime.datetime.now()

            if self.callback:
                self.callback(self.output)

            crew = self.agent.crew  # type: ignore[union-attr]
            if crew and crew.task_callback and crew.task_callback != self.callback:
                crew.task_callback(self.output)

            if self.output_file:
                content = (
                    json_output
                    if json_output
                    else (
                        pydantic_output.model_dump_json() if pydantic_output else result
                    )
                )
                self._save_file(content)
            crewai_event_bus.emit(
                self,
                TaskCompletedEvent(output=task_output, task=self),  # type: ignore[no-untyped-call]
            )
            return task_output
        except Exception as e:
            self.end_time = datetime.datetime.now()
            crewai_event_bus.emit(self, TaskFailedEvent(error=str(e), task=self))  # type: ignore[no-untyped-call]
            raise e  # Re-raise the exception after emitting the event
        finally:
            clear_task_files(self.id)

    def _execute_core(
        self,
        agent: BaseAgent | None,
        context: str | None,
        tools: list[Any] | None,
    ) -> TaskOutput:
        """Run the core execution logic of the task."""
        self._store_input_files()
        try:
            agent = agent or self.agent
            self.agent = agent
            if not agent:
                raise Exception(
                    f"The task '{self.description}' has no agent assigned, therefore it can't be executed directly and should be executed in a Crew using a specific process that support that, like hierarchical."
                )

            self.start_time = datetime.datetime.now()

            self.prompt_context = context
            tools = tools or self.tools or []

            self.processed_by_agents.add(agent.role)
            crewai_event_bus.emit(self, TaskStartedEvent(context=context, task=self))  # type: ignore[no-untyped-call]
            result = agent.execute_task(
                task=self,
                context=context,
                tools=tools,
            )

            if not self._guardrails and not self._guardrail:
                pydantic_output, json_output = self._export_output(result)
            else:
                pydantic_output, json_output = None, None

            task_output = TaskOutput(
                name=self.name or self.description,
                description=self.description,
                expected_output=self.expected_output,
                raw=result,
                pydantic=pydantic_output,
                json_dict=json_output,
                agent=agent.role,
                output_format=self._get_output_format(),
                messages=agent.last_messages,  # type: ignore[attr-defined]
            )

            if self._guardrails:
                for idx, guardrail in enumerate(self._guardrails):
                    task_output = self._invoke_guardrail_function(
                        task_output=task_output,
                        agent=agent,
                        tools=tools,
                        guardrail=guardrail,
                        guardrail_index=idx,
                    )

            # backwards support
            if self._guardrail:
                task_output = self._invoke_guardrail_function(
                    task_output=task_output,
                    agent=agent,
                    tools=tools,
                    guardrail=self._guardrail,
                )

            self.output = task_output
            self.end_time = datetime.datetime.now()

            if self.callback:
                self.callback(self.output)

            crew = self.agent.crew  # type: ignore[union-attr]
            if crew and crew.task_callback and crew.task_callback != self.callback:
                crew.task_callback(self.output)

            if self.output_file:
                content = (
                    json_output
                    if json_output
                    else (
                        pydantic_output.model_dump_json() if pydantic_output else result
                    )
                )
                self._save_file(content)
            crewai_event_bus.emit(
                self,
                TaskCompletedEvent(output=task_output, task=self),  # type: ignore[no-untyped-call]
            )
            return task_output
        except Exception as e:
            self.end_time = datetime.datetime.now()
            crewai_event_bus.emit(self, TaskFailedEvent(error=str(e), task=self))  # type: ignore[no-untyped-call]
            raise e  # Re-raise the exception after emitting the event
        finally:
            clear_task_files(self.id)

    def prompt(self) -> str:
        """Generates the task prompt with optional markdown formatting.

        When the markdown attribute is True, instructions for formatting the
        response in Markdown syntax will be added to the prompt.

        Returns:
            str: The formatted prompt string containing the task description,
                 expected output, and optional markdown formatting instructions.
        """
        description = self.description

        should_inject = self.allow_crewai_trigger_context

        if should_inject and self.agent:
            crew = getattr(self.agent, "crew", None)
            if crew and hasattr(crew, "_inputs") and crew._inputs:
                trigger_payload = crew._inputs.get("crewai_trigger_payload")
                if trigger_payload is not None:
                    description += f"\n\nTrigger Payload: {trigger_payload}"

        if self.agent and self.agent.crew:
            files = get_all_files(self.agent.crew.id, self.id)
            if files:
                supported_types: list[str] = []
                if self.agent.llm and self.agent.llm.supports_multimodal():
                    provider: str = str(
                        getattr(self.agent.llm, "provider", None)
                        or getattr(self.agent.llm, "model", "openai")
                    )
                    api: str | None = getattr(self.agent.llm, "api", None)
                    supported_types = get_supported_content_types(provider, api)

                def is_auto_injected(content_type: str) -> bool:
                    return any(content_type.startswith(t) for t in supported_types)

                auto_injected_files = {
                    name: f_input
                    for name, f_input in files.items()
                    if is_auto_injected(f_input.content_type)
                }
                tool_files = {
                    name: f_input
                    for name, f_input in files.items()
                    if not is_auto_injected(f_input.content_type)
                }

                file_lines: list[str] = []

                if auto_injected_files:
                    file_lines.append(
                        "Input files (content already loaded in conversation):"
                    )
                    for name, file_input in auto_injected_files.items():
                        filename = file_input.filename or name
                        file_lines.append(f'  - "{name}" ({filename})')

                if tool_files:
                    file_lines.append(
                        "Available input files (use the name in quotes with read_file tool):"
                    )
                    for name, file_input in tool_files.items():
                        filename = file_input.filename or name
                        content_type = file_input.content_type
                        file_lines.append(f'  - "{name}" ({filename}, {content_type})')

                if file_lines:
                    description += "\n\n" + "\n".join(file_lines)

        tasks_slices = [description]

        output = self.i18n.slice("expected_output").format(
            expected_output=self.expected_output
        )
        tasks_slices = [description, output]

        if self.markdown:
            markdown_instruction = """Your final answer MUST be formatted in Markdown syntax.
Follow these guidelines:
- Use # for headers
- Use ** for bold text
- Use * for italic text
- Use - or * for bullet points
- Use `code` for inline code
- Use ```language for code blocks"""
            tasks_slices.append(markdown_instruction)
        return "\n".join(tasks_slices)

    def interpolate_inputs_and_add_conversation_history(
        self, inputs: dict[str, str | int | float | dict[str, Any] | list[Any]]
    ) -> None:
        """Interpolate inputs into the task description, expected output, and output file path.
           Add conversation history if present.

        Args:
            inputs: Dictionary mapping template variables to their values.
                   Supported value types are strings, integers, and floats.

        Raises:
            ValueError: If a required template variable is missing from inputs.
        """
        if self._original_description is None:
            self._original_description = self.description
        if self._original_expected_output is None:
            self._original_expected_output = self.expected_output
        if self.output_file is not None and self._original_output_file is None:
            self._original_output_file = self.output_file

        if not inputs:
            return

        try:
            self.description = interpolate_only(
                input_string=self._original_description, inputs=inputs
            )
        except KeyError as e:
            raise ValueError(
                f"Missing required template variable '{e.args[0]}' in description"
            ) from e
        except ValueError as e:
            raise ValueError(f"Error interpolating description: {e!s}") from e

        try:
            self.expected_output = interpolate_only(
                input_string=self._original_expected_output, inputs=inputs
            )
        except (KeyError, ValueError) as e:
            raise ValueError(f"Error interpolating expected_output: {e!s}") from e

        if self.output_file is not None:
            try:
                self.output_file = interpolate_only(
                    input_string=self._original_output_file, inputs=inputs
                )
            except (KeyError, ValueError) as e:
                raise ValueError(f"Error interpolating output_file path: {e!s}") from e

        if inputs.get("crew_chat_messages"):
            conversation_instruction = self.i18n.slice(
                "conversation_history_instruction"
            )

            crew_chat_messages_json = str(inputs["crew_chat_messages"])

            try:
                crew_chat_messages = json.loads(crew_chat_messages_json)
            except json.JSONDecodeError as e:
                if self.agent and self.agent.verbose:
                    _printer.print(
                        f"An error occurred while parsing crew chat messages: {e}",
                        color="red",
                    )
                raise

            conversation_history = "\n".join(
                f"{msg['role'].capitalize()}: {msg['content']}"
                for msg in crew_chat_messages
                if isinstance(msg, dict) and "role" in msg and "content" in msg
            )

            self.description += (
                f"\n\n{conversation_instruction}\n\n{conversation_history}"
            )

    def increment_tools_errors(self) -> None:
        """Increment the tools errors counter."""
        self.tools_errors += 1

    def increment_delegations(self, agent_name: str | None) -> None:
        """Increment the delegations counter."""
        if agent_name:
            self.processed_by_agents.add(agent_name)
        self.delegations += 1

    def copy(  # type: ignore
        self, agents: list[BaseAgent], task_mapping: dict[str, Task]
    ) -> Task:
        """Creates a deep copy of the Task while preserving its original class type.

        Args:
            agents: List of agents available for the task.
            task_mapping: Dictionary mapping task IDs to Task instances.

        Returns:
            A copy of the task with the same class type as the original.
        """
        exclude = {
            "id",
            "agent",
            "context",
            "tools",
        }

        copied_data = self.model_dump(exclude=exclude)
        copied_data = {k: v for k, v in copied_data.items() if v is not None}

        cloned_context = (
            self.context
            if self.context is NOT_SPECIFIED
            else [task_mapping[context_task.key] for context_task in self.context]
            if isinstance(self.context, list)
            else None
        )

        def get_agent_by_role(role: str) -> BaseAgent | None:
            return next((agent for agent in agents if agent.role == role), None)

        cloned_agent = get_agent_by_role(self.agent.role) if self.agent else None
        cloned_tools = shallow_copy(self.tools) if self.tools else []

        return self.__class__(
            **copied_data,
            context=cloned_context,
            agent=cloned_agent,
            tools=cloned_tools,
        )

    def _export_output(
        self, result: str
    ) -> tuple[BaseModel | None, dict[str, Any] | None]:
        pydantic_output: BaseModel | None = None
        json_output: dict[str, Any] | None = None

        if self.output_pydantic or self.output_json:
            model_output = convert_to_model(
                result,
                self.output_pydantic,
                self.output_json,
                self.agent,
                self.converter_cls,
            )

            if isinstance(model_output, BaseModel):
                pydantic_output = model_output
            elif isinstance(model_output, dict):
                json_output = model_output
            elif isinstance(model_output, str):
                try:
                    json_output = json.loads(model_output)
                except json.JSONDecodeError:
                    json_output = None

        return pydantic_output, json_output

    def _get_output_format(self) -> OutputFormat:
        if self.output_json:
            return OutputFormat.JSON
        if self.output_pydantic:
            return OutputFormat.PYDANTIC
        return OutputFormat.RAW

    def _save_file(self, result: dict[str, Any] | str | Any) -> None:
        """Save task output to a file.

        Note:
            For cross-platform file writing, especially on Windows, consider using FileWriterTool
            from the crewai_tools package:
                pip install 'crewai[tools]'
                from crewai_tools import FileWriterTool

        Args:
            result: The result to save to the file. Can be a dict or any stringifiable object.

        Raises:
            ValueError: If output_file is not set
            RuntimeError: If there is an error writing to the file. For cross-platform
                compatibility, especially on Windows, use FileWriterTool from crewai_tools
                package.
        """
        if self.output_file is None:
            raise ValueError("output_file is not set.")

        filewriter_recommendation = (
            "For cross-platform file writing, especially on Windows, "
            "use FileWriterTool from crewai_tools package."
        )

        try:
            resolved_path = Path(self.output_file).expanduser().resolve()
            directory = resolved_path.parent

            if self.create_directory and not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
            elif not self.create_directory and not directory.exists():
                raise RuntimeError(
                    f"Directory {directory} does not exist and create_directory is False"
                )

            with resolved_path.open("w", encoding="utf-8") as file:
                if isinstance(result, dict):
                    import json

                    json.dump(result, file, ensure_ascii=False, indent=2)
                else:
                    file.write(str(result))
        except (OSError, IOError) as e:
            raise RuntimeError(
                "\n".join(
                    [f"Failed to save output file: {e}", filewriter_recommendation]
                )
            ) from e
        return

    def _store_input_files(self) -> None:
        """Store task input files in the file store."""
        if not HAS_CREWAI_FILES or not self.input_files:
            return

        store_task_files(self.id, self.input_files)

    def __repr__(self) -> str:
        return f"Task(description={self.description}, expected_output={self.expected_output})"

    @property
    def fingerprint(self) -> Fingerprint:
        """Get the fingerprint of the task.

        Returns:
            Fingerprint: The fingerprint of the task
        """
        return self.security_config.fingerprint

    def _invoke_guardrail_function(
        self,
        task_output: TaskOutput,
        agent: BaseAgent,
        tools: list[BaseTool],
        guardrail: GuardrailCallable | None,
        guardrail_index: int | None = None,
    ) -> TaskOutput:
        if not guardrail:
            return task_output

        if guardrail_index is not None:
            current_retry_count = self._guardrail_retry_counts.get(guardrail_index, 0)
        else:
            current_retry_count = self.retry_count

        max_attempts = self.guardrail_max_retries + 1

        for attempt in range(max_attempts):
            guardrail_result = process_guardrail(
                output=task_output,
                guardrail=guardrail,
                retry_count=current_retry_count,
                event_source=self,
                from_task=self,
                from_agent=agent,
            )

            if guardrail_result.success:
                # Guardrail passed
                if guardrail_result.result is None:
                    raise Exception(
                        "Task guardrail returned None as result. This is not allowed."
                    )

                if isinstance(guardrail_result.result, str):
                    task_output.raw = guardrail_result.result
                    pydantic_output, json_output = self._export_output(
                        guardrail_result.result
                    )
                    task_output.pydantic = pydantic_output
                    task_output.json_dict = json_output
                elif isinstance(guardrail_result.result, TaskOutput):
                    task_output = guardrail_result.result

                return task_output

            # Guardrail failed
            if attempt >= self.guardrail_max_retries:
                # Max retries reached
                guardrail_name = (
                    f"guardrail {guardrail_index}"
                    if guardrail_index is not None
                    else "guardrail"
                )
                raise Exception(
                    f"Task failed {guardrail_name} validation after {self.guardrail_max_retries} retries. "
                    f"Last error: {guardrail_result.error}"
                )

            if guardrail_index is not None:
                current_retry_count += 1
                self._guardrail_retry_counts[guardrail_index] = current_retry_count
            else:
                self.retry_count += 1
                current_retry_count = self.retry_count

            context = self.i18n.errors("validation_error").format(
                guardrail_result_error=guardrail_result.error,
                task_output=task_output.raw,
            )
            if agent and agent.verbose:
                printer = Printer()
                printer.print(
                    content=f"Guardrail {guardrail_index if guardrail_index is not None else ''} blocked (attempt {attempt + 1}/{max_attempts}), retrying due to: {guardrail_result.error}\n",
                    color="yellow",
                )

            # Regenerate output from agent
            result = agent.execute_task(
                task=self,
                context=context,
                tools=tools,
            )

            pydantic_output, json_output = self._export_output(result)
            task_output = TaskOutput(
                name=self.name or self.description,
                description=self.description,
                expected_output=self.expected_output,
                raw=result,
                pydantic=pydantic_output,
                json_dict=json_output,
                agent=agent.role,
                output_format=self._get_output_format(),
                messages=agent.last_messages,  # type: ignore[attr-defined]
            )

        return task_output

    async def _ainvoke_guardrail_function(
        self,
        task_output: TaskOutput,
        agent: BaseAgent,
        tools: list[BaseTool],
        guardrail: GuardrailCallable | None,
        guardrail_index: int | None = None,
    ) -> TaskOutput:
        """Invoke the guardrail function asynchronously."""
        if not guardrail:
            return task_output

        if guardrail_index is not None:
            current_retry_count = self._guardrail_retry_counts.get(guardrail_index, 0)
        else:
            current_retry_count = self.retry_count

        max_attempts = self.guardrail_max_retries + 1

        for attempt in range(max_attempts):
            guardrail_result = process_guardrail(
                output=task_output,
                guardrail=guardrail,
                retry_count=current_retry_count,
                event_source=self,
                from_task=self,
                from_agent=agent,
            )

            if guardrail_result.success:
                if guardrail_result.result is None:
                    raise Exception(
                        "Task guardrail returned None as result. This is not allowed."
                    )

                if isinstance(guardrail_result.result, str):
                    task_output.raw = guardrail_result.result
                    pydantic_output, json_output = self._export_output(
                        guardrail_result.result
                    )
                    task_output.pydantic = pydantic_output
                    task_output.json_dict = json_output
                elif isinstance(guardrail_result.result, TaskOutput):
                    task_output = guardrail_result.result

                return task_output

            if attempt >= self.guardrail_max_retries:
                guardrail_name = (
                    f"guardrail {guardrail_index}"
                    if guardrail_index is not None
                    else "guardrail"
                )
                raise Exception(
                    f"Task failed {guardrail_name} validation after {self.guardrail_max_retries} retries. "
                    f"Last error: {guardrail_result.error}"
                )

            if guardrail_index is not None:
                current_retry_count += 1
                self._guardrail_retry_counts[guardrail_index] = current_retry_count
            else:
                self.retry_count += 1
                current_retry_count = self.retry_count

            context = self.i18n.errors("validation_error").format(
                guardrail_result_error=guardrail_result.error,
                task_output=task_output.raw,
            )
            if agent and agent.verbose:
                printer = Printer()
                printer.print(
                    content=f"Guardrail {guardrail_index if guardrail_index is not None else ''} blocked (attempt {attempt + 1}/{max_attempts}), retrying due to: {guardrail_result.error}\n",
                    color="yellow",
                )

            result = await agent.aexecute_task(
                task=self,
                context=context,
                tools=tools,
            )

            pydantic_output, json_output = self._export_output(result)
            task_output = TaskOutput(
                name=self.name or self.description,
                description=self.description,
                expected_output=self.expected_output,
                raw=result,
                pydantic=pydantic_output,
                json_dict=json_output,
                agent=agent.role,
                output_format=self._get_output_format(),
                messages=agent.last_messages,  # type: ignore[attr-defined]
            )

        return task_output
