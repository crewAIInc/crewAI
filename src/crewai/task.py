import datetime
import inspect
import json
import logging
import threading
import uuid
from concurrent.futures import Future
from copy import copy
from hashlib import md5
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

from opentelemetry.trace import Span
from pydantic import (
    UUID4,
    BaseModel,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.tasks.guardrail_result import GuardrailResult
from crewai.tasks.output_format import OutputFormat
from crewai.tasks.task_output import TaskOutput
from crewai.telemetry.telemetry import Telemetry
from crewai.tools.base_tool import BaseTool
from crewai.utilities.config import process_config
from crewai.utilities.converter import Converter, convert_to_model
from crewai.utilities.i18n import I18N
from crewai.utilities.printer import Printer


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
        output_json: Pydantic model for structuring JSON output.
        output_pydantic: Pydantic model for task output.
        tools: List of tools/resources limited for task execution.
    """

    __hash__ = object.__hash__  # type: ignore
    logger: ClassVar[logging.Logger] = logging.getLogger(__name__)
    used_tools: int = 0
    tools_errors: int = 0
    delegations: int = 0
    i18n: I18N = I18N()
    name: Optional[str] = Field(default=None)
    prompt_context: Optional[str] = None
    description: str = Field(description="Description of the actual task.")
    expected_output: str = Field(
        description="Clear definition of expected output for the task."
    )
    config: Optional[Dict[str, Any]] = Field(
        description="Configuration for the agent",
        default=None,
    )
    callback: Optional[Any] = Field(
        description="Callback to be executed after the task is completed.", default=None
    )
    agent: Optional[BaseAgent] = Field(
        description="Agent responsible for execution the task.", default=None
    )
    context: Optional[List["Task"]] = Field(
        description="Other tasks that will have their output used as context for this task.",
        default=None,
    )
    async_execution: Optional[bool] = Field(
        description="Whether the task should be executed asynchronously or not.",
        default=False,
    )
    output_json: Optional[Type[BaseModel]] = Field(
        description="A Pydantic model to be used to create a JSON output.",
        default=None,
    )
    output_pydantic: Optional[Type[BaseModel]] = Field(
        description="A Pydantic model to be used to create a Pydantic output.",
        default=None,
    )
    output_file: Optional[str] = Field(
        description="A file path to be used to create a file output.",
        default=None,
    )
    output: Optional[TaskOutput] = Field(
        description="Task output, it's final result after being executed", default=None
    )
    tools: Optional[List[BaseTool]] = Field(
        default_factory=list,
        description="Tools the agent is limited to use for this task.",
    )
    id: UUID4 = Field(
        default_factory=uuid.uuid4,
        frozen=True,
        description="Unique identifier for the object, not set by user.",
    )
    human_input: Optional[bool] = Field(
        description="Whether the task should have a human review the final answer of the agent",
        default=False,
    )
    converter_cls: Optional[Type[Converter]] = Field(
        description="A converter class used to export structured output",
        default=None,
    )
    processed_by_agents: Set[str] = Field(default_factory=set)
    guardrail: Optional[Callable[[TaskOutput], Tuple[bool, Any]]] = Field(
        default=None,
        description="Function to validate task output before proceeding to next task",
    )
    max_retries: int = Field(
        default=3, description="Maximum number of retries when guardrail fails"
    )
    retry_count: int = Field(default=0, description="Current number of retries")
    start_time: Optional[datetime.datetime] = Field(
        default=None, description="Start time of the task execution"
    )
    end_time: Optional[datetime.datetime] = Field(
        default=None, description="End time of the task execution"
    )

    @field_validator("guardrail")
    @classmethod
    def validate_guardrail_function(cls, v: Optional[Callable]) -> Optional[Callable]:
        """Validate that the guardrail function has the correct signature and behavior.

        While type hints provide static checking, this validator ensures runtime safety by:
        1. Verifying the function accepts exactly one parameter (the TaskOutput)
        2. Checking return type annotations match Tuple[bool, Any] if present
        3. Providing clear, immediate error messages for debugging

        This runtime validation is crucial because:
        - Type hints are optional and can be ignored at runtime
        - Function signatures need immediate validation before task execution
        - Clear error messages help users debug guardrail implementation issues

        Args:
            v: The guardrail function to validate

        Returns:
            The validated guardrail function

        Raises:
            ValueError: If the function signature is invalid or return annotation
                       doesn't match Tuple[bool, Any]
        """
        if v is not None:
            sig = inspect.signature(v)
            if len(sig.parameters) != 1:
                raise ValueError("Guardrail function must accept exactly one parameter")

            # Check return annotation if present, but don't require it
            return_annotation = sig.return_annotation
            if return_annotation != inspect.Signature.empty:
                if not (
                    return_annotation == Tuple[bool, Any]
                    or str(return_annotation) == "Tuple[bool, Any]"
                ):
                    raise ValueError(
                        "If return type is annotated, it must be Tuple[bool, Any]"
                    )
        return v

    _telemetry: Telemetry = PrivateAttr(default_factory=Telemetry)
    _execution_span: Optional[Span] = PrivateAttr(default=None)
    _original_description: Optional[str] = PrivateAttr(default=None)
    _original_expected_output: Optional[str] = PrivateAttr(default=None)
    _original_output_file: Optional[str] = PrivateAttr(default=None)
    _thread: Optional[threading.Thread] = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def process_model_config(cls, values):
        return process_config(values, cls)

    @model_validator(mode="after")
    def validate_required_fields(self):
        required_fields = ["description", "expected_output"]
        for field in required_fields:
            if getattr(self, field) is None:
                raise ValueError(
                    f"{field} must be provided either directly or through config"
                )
        return self

    @field_validator("id", mode="before")
    @classmethod
    def _deny_user_set_id(cls, v: Optional[UUID4]) -> None:
        if v:
            raise PydanticCustomError(
                "may_not_set_field", "This field is not to be set by the user.", {}
            )

    @field_validator("output_file")
    @classmethod
    def output_file_validation(cls, value: Optional[str]) -> Optional[str]:
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
        if value.startswith("~") or value.startswith("$"):
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
    def set_attributes_based_on_config(self) -> "Task":
        """Set attributes based on the agent configuration."""
        if self.config:
            for key, value in self.config.items():
                setattr(self, key, value)
        return self

    @model_validator(mode="after")
    def check_tools(self):
        """Check if the tools are set."""
        if not self.tools and self.agent and self.agent.tools:
            self.tools.extend(self.agent.tools)
        return self

    @model_validator(mode="after")
    def check_output(self):
        """Check if an output type is set."""
        output_types = [self.output_json, self.output_pydantic]
        if len([type for type in output_types if type]) > 1:
            raise PydanticCustomError(
                "output_type",
                "Only one output type can be set, either output_pydantic or output_json.",
                {},
            )
        return self

    def execute_sync(
        self,
        agent: Optional[BaseAgent] = None,
        context: Optional[str] = None,
        tools: Optional[List[BaseTool]] = None,
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
        context: Optional[str] = None,
        tools: Optional[List[BaseTool]] = None,
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
        agent: Optional[BaseAgent],
        context: Optional[str],
        tools: Optional[List[Any]],
        future: Future[TaskOutput],
    ) -> None:
        """Execute the task asynchronously with context handling."""
        result = self._execute_core(agent, context, tools)
        future.set_result(result)

    def _execute_core(
        self,
        agent: Optional[BaseAgent],
        context: Optional[str],
        tools: Optional[List[Any]],
    ) -> TaskOutput:
        """Run the core execution logic of the task."""
        agent = agent or self.agent
        self.agent = agent
        if not agent:
            raise Exception(
                f"The task '{self.description}' has no agent assigned, therefore it can't be executed directly and should be executed in a Crew using a specific process that support that, like hierarchical."
            )

        self.start_time = datetime.datetime.now()
        self._execution_span = self._telemetry.task_started(crew=agent.crew, task=self)

        self.prompt_context = context
        tools = tools or self.tools or []

        self.processed_by_agents.add(agent.role)

        result = agent.execute_task(
            task=self,
            context=context,
            tools=tools,
        )

        pydantic_output, json_output = self._export_output(result)
        task_output = TaskOutput(
            name=self.name,
            description=self.description,
            expected_output=self.expected_output,
            raw=result,
            pydantic=pydantic_output,
            json_dict=json_output,
            agent=agent.role,
            output_format=self._get_output_format(),
        )

        if self.guardrail:
            guardrail_result = GuardrailResult.from_tuple(self.guardrail(task_output))
            if not guardrail_result.success:
                if self.retry_count >= self.max_retries:
                    raise Exception(
                        f"Task failed guardrail validation after {self.max_retries} retries. "
                        f"Last error: {guardrail_result.error}"
                    )

                self.retry_count += 1
                context = self.i18n.errors("validation_error").format(
                    guardrail_result_error=guardrail_result.error,
                    task_output=task_output.raw
                )
                printer = Printer()
                printer.print(
                    content=f"Guardrail blocked, retrying, due to:{guardrail_result.error}\n",
                    color="yellow",
                )
                return self._execute_core(agent, context, tools)

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

        self.output = task_output
        self.end_time = datetime.datetime.now()

        if self.callback:
            self.callback(self.output)

        if self._execution_span:
            self._telemetry.task_ended(self._execution_span, self, agent.crew)
            self._execution_span = None

        if self.output_file:
            content = (
                json_output
                if json_output
                else pydantic_output.model_dump_json()
                if pydantic_output
                else result
            )
            self._save_file(content)

        return task_output

    def prompt(self) -> str:
        """Prompt the task.

        Returns:
            Prompt of the task.
        """
        tasks_slices = [self.description]

        output = self.i18n.slice("expected_output").format(
            expected_output=self.expected_output
        )
        tasks_slices = [self.description, output]
        return "\n".join(tasks_slices)

    def interpolate_inputs(self, inputs: Dict[str, Union[str, int, float]]) -> None:
        """Interpolate inputs into the task description, expected output, and output file path.

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
            self.description = self._original_description.format(**inputs)
        except KeyError as e:
            raise ValueError(
                f"Missing required template variable '{e.args[0]}' in description"
            ) from e
        except ValueError as e:
            raise ValueError(f"Error interpolating description: {str(e)}") from e

        try:
            self.expected_output = self.interpolate_only(
                input_string=self._original_expected_output, inputs=inputs
            )
        except (KeyError, ValueError) as e:
            raise ValueError(f"Error interpolating expected_output: {str(e)}") from e

        if self.output_file is not None:
            try:
                self.output_file = self.interpolate_only(
                    input_string=self._original_output_file, inputs=inputs
                )
            except (KeyError, ValueError) as e:
                raise ValueError(
                    f"Error interpolating output_file path: {str(e)}"
                ) from e

    def interpolate_only(
        self, input_string: Optional[str], inputs: Dict[str, Union[str, int, float]]
    ) -> str:
        """Interpolate placeholders (e.g., {key}) in a string while leaving JSON untouched.

        Args:
            input_string: The string containing template variables to interpolate.
                         Can be None or empty, in which case an empty string is returned.
            inputs: Dictionary mapping template variables to their values.
                   Supported value types are strings, integers, and floats.
                   If input_string is empty or has no placeholders, inputs can be empty.

        Returns:
            The interpolated string with all template variables replaced with their values.
            Empty string if input_string is None or empty.

        Raises:
            ValueError: If a required template variable is missing from inputs.
            KeyError: If a template variable is not found in the inputs dictionary.
        """
        if input_string is None or not input_string:
            return ""
        if "{" not in input_string and "}" not in input_string:
            return input_string
        if not inputs:
            raise ValueError(
                "Inputs dictionary cannot be empty when interpolating variables"
            )

        try:
            # Validate input types
            for key, value in inputs.items():
                if not isinstance(value, (str, int, float)):
                    raise ValueError(
                        f"Value for key '{key}' must be a string, integer, or float, got {type(value).__name__}"
                    )

            escaped_string = input_string.replace("{", "{{").replace("}", "}}")

            for key in inputs.keys():
                escaped_string = escaped_string.replace(f"{{{{{key}}}}}", f"{{{key}}}")

            return escaped_string.format(**inputs)
        except KeyError as e:
            raise KeyError(
                f"Template variable '{e.args[0]}' not found in inputs dictionary"
            ) from e
        except ValueError as e:
            raise ValueError(f"Error during string interpolation: {str(e)}") from e

    def increment_tools_errors(self) -> None:
        """Increment the tools errors counter."""
        self.tools_errors += 1

    def increment_delegations(self, agent_name: Optional[str]) -> None:
        """Increment the delegations counter."""
        if agent_name:
            self.processed_by_agents.add(agent_name)
        self.delegations += 1

    def copy(
        self, agents: List["BaseAgent"], task_mapping: Dict[str, "Task"]
    ) -> "Task":
        """Create a deep copy of the Task."""
        exclude = {
            "id",
            "agent",
            "context",
            "tools",
        }

        copied_data = self.model_dump(exclude=exclude)
        copied_data = {k: v for k, v in copied_data.items() if v is not None}

        cloned_context = (
            [task_mapping[context_task.key] for context_task in self.context]
            if self.context
            else None
        )

        def get_agent_by_role(role: str) -> Union["BaseAgent", None]:
            return next((agent for agent in agents if agent.role == role), None)

        cloned_agent = get_agent_by_role(self.agent.role) if self.agent else None
        cloned_tools = copy(self.tools) if self.tools else []

        copied_task = Task(
            **copied_data,
            context=cloned_context,
            agent=cloned_agent,
            tools=cloned_tools,
        )

        return copied_task

    def _export_output(
        self, result: str
    ) -> Tuple[Optional[BaseModel], Optional[Dict[str, Any]]]:
        pydantic_output: Optional[BaseModel] = None
        json_output: Optional[Dict[str, Any]] = None

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

    def _save_file(self, result: Any) -> None:
        """Save task output to a file.

        Args:
            result: The result to save to the file. Can be a dict or any stringifiable object.

        Raises:
            ValueError: If output_file is not set
            RuntimeError: If there is an error writing to the file
        """
        if self.output_file is None:
            raise ValueError("output_file is not set.")

        try:
            resolved_path = Path(self.output_file).expanduser().resolve()
            directory = resolved_path.parent

            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)

            with resolved_path.open("w", encoding="utf-8") as file:
                if isinstance(result, dict):
                    import json

                    json.dump(result, file, ensure_ascii=False, indent=2)
                else:
                    file.write(str(result))
        except (OSError, IOError) as e:
            raise RuntimeError(f"Failed to save output file: {e}")
        return None

    def __repr__(self):
        return f"Task(description={self.description}, expected_output={self.expected_output})"
