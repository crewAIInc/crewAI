import os
import re
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from copy import copy
from typing import Any, Dict, List, Optional, Type, Union

from langchain_openai import ChatOpenAI
from opentelemetry.trace import Span
from pydantic import UUID4, BaseModel, Field, field_validator, model_validator
from pydantic_core import PydanticCustomError

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.tasks.task_output import TaskOutput
from crewai.telemetry.telemetry import Telemetry
from crewai.utilities.converter import ConverterError
from crewai.utilities.converter import Converter
from crewai.utilities.i18n import I18N
from crewai.utilities.printer import Printer
from crewai.utilities.pydantic_schema_parser import PydanticSchemaParser


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

    class Config:
        arbitrary_types_allowed = True

    __hash__ = object.__hash__  # type: ignore
    used_tools: int = 0
    tools_errors: int = 0
    delegations: int = 0
    i18n: I18N = I18N()
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
    tools: Optional[List[Any]] = Field(
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

    _telemetry: Telemetry
    _execution_span: Span | None = None
    _original_description: str | None = None
    _original_expected_output: str | None = None
    _future: Future | None = None

    def __init__(__pydantic_self__, **data):
        config = data.pop("config", {})
        super().__init__(**config, **data)

    @field_validator("id", mode="before")
    @classmethod
    def _deny_user_set_id(cls, v: Optional[UUID4]) -> None:
        if v:
            raise PydanticCustomError(
                "may_not_set_field", "This field is not to be set by the user.", {}
            )

    @field_validator("output_file")
    @classmethod
    def output_file_validattion(cls, value: str) -> str:
        """Validate the output file path by removing the / from the beginning of the path."""
        if value.startswith("/"):
            return value[1:]
        return value

    @model_validator(mode="after")
    def set_private_attrs(self) -> "Task":
        """Set private attributes."""
        self._telemetry = Telemetry()
        return self

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

    def wait_for_completion(self) -> str | BaseModel:
        """Wait for asynchronous task completion and return the output."""
        assert self.async_execution, "Task is not set to be executed asynchronously."

        if self._future:
            self._future.result()  # Wait for the future to complete
            self._future = None

        assert self.output, "Task output is not set."

        return self.output.exported_output

    def execute(
        self,
        agent: BaseAgent | None = None,
        context: Optional[str] = None,
        tools: Optional[List[Any]] = None,
    ) -> str | None:
        """Execute the task.

        Returns:
            Output of the task.
        """

        self._execution_span = self._telemetry.task_started(self)

        agent = agent or self.agent
        if not agent:
            raise Exception(
                f"The task '{self.description}' has no agent assigned, therefore it can't be executed directly "
                "and should be executed in a Crew using a specific process that support that, like hierarchical."
            )

        if self.context:
            internal_context = []
            for task in self.context:
                if task.async_execution:
                    task.wait_for_completion()
                if task.output:
                    internal_context.append(task.output.raw_output)
            context = "\n".join(internal_context)

        self.prompt_context = context
        tools = tools or self.tools

        if self.async_execution:
            with ThreadPoolExecutor() as executor:
                self._future = executor.submit(
                    self._execute, agent, self, context, tools
                )
            return None
        else:
            result = self._execute(
                task=self,
                agent=agent,
                context=context,
                tools=tools,
            )
            return result

    def _execute(self, agent: "BaseAgent", task, context, tools) -> str | None:
        result = agent.execute_task(
            task=task,
            context=context,
            tools=tools,
        )
        exported_output = self._export_output(result)

        self.output = TaskOutput(
            description=self.description,
            exported_output=exported_output,
            raw_output=result,
            agent=agent.role,
        )

        if self.callback:
            self.callback(self.output)

        if self._execution_span:
            self._telemetry.task_ended(self._execution_span, self)
            self._execution_span = None

        return exported_output

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

    def interpolate_inputs(self, inputs: Dict[str, Any]) -> None:
        """Interpolate inputs into the task description and expected output."""
        if self._original_description is None:
            self._original_description = self.description
        if self._original_expected_output is None:
            self._original_expected_output = self.expected_output

        if inputs:
            self.description = self._original_description.format(**inputs)
            self.expected_output = self._original_expected_output.format(**inputs)

    def increment_tools_errors(self) -> None:
        """Increment the tools errors counter."""
        self.tools_errors += 1

    def increment_delegations(self) -> None:
        """Increment the delegations counter."""
        self.delegations += 1

    def copy(self, agents: Optional[List["BaseAgent"]] = None) -> "Task":  # type: ignore # Signature of "copy" incompatible with supertype "BaseModel"
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
            [task.copy() for task in self.context] if self.context else None
        )

        def get_agent_by_role(role: str) -> Union["BaseAgent", None]:
            return next((agent for agent in agents if agent.role == role), None)  # type: ignore # Item "None" of "list[BaseAgent] | None" has no attribute "__iter__" (not iterable)

        cloned_agent = get_agent_by_role(self.agent.role) if self.agent else None
        cloned_tools = copy(self.tools) if self.tools else []

        copied_task = Task(
            **copied_data,
            context=cloned_context,
            agent=cloned_agent,
            tools=cloned_tools,
        )

        return copied_task

    def _create_converter(self, *args, **kwargs) -> Converter: # type: ignore
      converter = self.agent.get_output_converter(  # type: ignore # Item "None" of "BaseAgent | None" has no attribute "get_output_converter"
        *args, **kwargs
      )
      if self.converter_cls:
        converter = self.converter_cls(  # type: ignore # Item "None" of "BaseAgent | None" has no attribute "get_output_converter"
          *args, **kwargs
        )          
      return converter

    def _export_output(self, result: str) -> Any:
        exported_result = result
        instructions = "I'm gonna convert this raw text into valid JSON."

        if self.output_pydantic or self.output_json:
            model = self.output_pydantic or self.output_json

            # try to convert task_output directly to pydantic/json
            try:
                exported_result = model.model_validate_json(result)  # type: ignore # Item "None" of "type[BaseModel] | None" has no attribute "model_validate_json"
                if self.output_json:
                    return exported_result.model_dump()  # type: ignore # "str" has no attribute "model_dump"
                return exported_result
            except Exception:
                # sometimes the response contains valid JSON in the middle of text
                match = re.search(r"({.*})", result, re.DOTALL)
                if match:
                    try:
                        exported_result = model.model_validate_json(match.group(0))  # type: ignore # Item "None" of "type[BaseModel] | None" has no attribute "model_validate_json"
                        if self.output_json:
                            return exported_result.model_dump()  # type: ignore # "str" has no attribute "model_dump"
                        return exported_result
                    except Exception:
                        pass

            llm = getattr(self.agent, "function_calling_llm", None) or self.agent.llm  # type: ignore # Item "None" of "BaseAgent | None" has no attribute "function_calling_llm"
            if not self._is_gpt(llm):
                model_schema = PydanticSchemaParser(model=model).get_schema()  # type: ignore # Argument "model" to "PydanticSchemaParser" has incompatible type "type[BaseModel] | None"; expected "type[BaseModel]"
                instructions = f"{instructions}\n\nThe json should have the following structure, with the following keys:\n{model_schema}"

            converter = self._create_converter( # type: ignore # Item "None" of "BaseAgent | None" has no attribute "get_output_converter"
                llm=llm, text=result, model=model, instructions=instructions
            )

            if self.output_pydantic:
                exported_result = converter.to_pydantic()
            elif self.output_json:
                exported_result = converter.to_json()

            if isinstance(exported_result, ConverterError):
                Printer().print(
                    content=f"{exported_result.message} Using raw output instead.",
                    color="red",
                )
                exported_result = result

        if self.output_file:
            content = (
                exported_result
                if not self.output_pydantic
                else exported_result.model_dump_json()  # type: ignore # "str" has no attribute "json"
            )
            self._save_file(content)

        return exported_result

    def _is_gpt(self, llm) -> bool:
        return isinstance(llm, ChatOpenAI) and llm.openai_api_base is None

    def _save_file(self, result: Any) -> None:
        directory = os.path.dirname(self.output_file)  # type: ignore # Value of type variable "AnyOrLiteralStr" of "dirname" cannot be "str | None"

        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        with open(self.output_file, "w", encoding="utf-8") as file:  # type: ignore # Argument 1 to "open" has incompatible type "str | None"; expected "int | str | bytes | PathLike[str] | PathLike[bytes]"
            file.write(result)
        return None

    def __repr__(self):
        return f"Task(description={self.description}, expected_output={self.expected_output})"
