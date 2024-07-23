import json
import os
import re
import threading
import uuid
from concurrent.futures import Future
from copy import copy
from hashlib import md5
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from langchain_openai import ChatOpenAI
from opentelemetry.trace import Span
from pydantic import UUID4, BaseModel, Field, field_validator, model_validator
from pydantic_core import PydanticCustomError

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.tasks.output_format import OutputFormat
from crewai.tasks.task_output import TaskOutput
from crewai.telemetry.telemetry import Telemetry
from crewai.utilities.converter import Converter, ConverterError
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
    _thread: threading.Thread | None = None

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

    def execute_sync(
        self,
        agent: Optional[BaseAgent] = None,
        context: Optional[str] = None,
        tools: Optional[List[Any]] = None,
    ) -> TaskOutput:
        """Execute the task synchronously."""
        return self._execute_core(agent, context, tools)

    @property
    def key(self) -> str:
        description = self._original_description or self.description
        expected_output = self._original_expected_output or self.expected_output
        source = [description, expected_output]

        return md5("|".join(source).encode()).hexdigest()

    def execute_async(
        self,
        agent: BaseAgent | None = None,
        context: Optional[str] = None,
        tools: Optional[List[Any]] = None,
    ) -> Future[TaskOutput]:
        """Execute the task asynchronously."""
        future: Future[TaskOutput] = Future()
        threading.Thread(
            target=self._execute_task_async, args=(agent, context, tools, future)
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

        self._execution_span = self._telemetry.task_started(crew=agent.crew, task=self)

        self.prompt_context = context
        tools = tools or self.tools or []

        result = agent.execute_task(
            task=self,
            context=context,
            tools=tools,
        )

        pydantic_output, json_output = self._export_output(result)

        task_output = TaskOutput(
            description=self.description,
            raw=result,
            pydantic=pydantic_output,
            json_dict=json_output,
            agent=agent.role,
            output_format=self._get_output_format(),
        )
        self.output = task_output

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

    def copy(self, agents: List["BaseAgent"]) -> "Task":
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
            [task.copy(agents) for task in self.context] if self.context else None
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

    def _create_converter(self, *args, **kwargs) -> Converter:
        """Create a converter instance."""
        if self.agent and not self.converter_cls:
            converter = self.agent.get_output_converter(*args, **kwargs)
        elif self.converter_cls:
            converter = self.converter_cls(*args, **kwargs)

        if not converter:
            raise Exception("No output converter found or set.")

        return converter

    def _export_output(
        self, result: str
    ) -> Tuple[Optional[BaseModel], Optional[Dict[str, Any]]]:
        pydantic_output: Optional[BaseModel] = None
        json_output: Optional[Dict[str, Any]] = None

        if self.output_pydantic or self.output_json:
            model_output = self._convert_to_model(result)
            pydantic_output = (
                model_output if isinstance(model_output, BaseModel) else None
            )
            if isinstance(model_output, str):
                try:
                    json_output = json.loads(model_output)
                except json.JSONDecodeError:
                    json_output = None
            else:
                json_output = model_output if isinstance(model_output, dict) else None

        return pydantic_output, json_output

    def _convert_to_model(self, result: str) -> Union[dict, BaseModel, str]:
        model = self.output_pydantic or self.output_json
        if model is None:
            return result

        try:
            return self._validate_model(result, model)
        except Exception:
            return self._handle_partial_json(result, model)

    def _validate_model(
        self, result: str, model: Type[BaseModel]
    ) -> Union[dict, BaseModel]:
        exported_result = model.model_validate_json(result)
        if self.output_json:
            return exported_result.model_dump()
        return exported_result

    def _handle_partial_json(
        self, result: str, model: Type[BaseModel]
    ) -> Union[dict, BaseModel, str]:
        match = re.search(r"({.*})", result, re.DOTALL)
        if match:
            try:
                exported_result = model.model_validate_json(match.group(0))
                if self.output_json:
                    return exported_result.model_dump()
                return exported_result
            except Exception:
                pass

        return self._convert_with_instructions(result, model)

    def _convert_with_instructions(
        self, result: str, model: Type[BaseModel]
    ) -> Union[dict, BaseModel, str]:
        llm = self.agent.function_calling_llm or self.agent.llm  # type: ignore # Item "None" of "BaseAgent | None" has no attribute "function_calling_llm"
        instructions = self._get_conversion_instructions(model, llm)

        converter = self._create_converter(
            llm=llm, text=result, model=model, instructions=instructions
        )
        exported_result = (
            converter.to_pydantic() if self.output_pydantic else converter.to_json()
        )

        if isinstance(exported_result, ConverterError):
            Printer().print(
                content=f"{exported_result.message} Using raw output instead.",
                color="red",
            )
            return result

        return exported_result

    def _get_output_format(self) -> OutputFormat:
        if self.output_json:
            return OutputFormat.JSON
        if self.output_pydantic:
            return OutputFormat.PYDANTIC
        return OutputFormat.RAW

    def _get_conversion_instructions(self, model: Type[BaseModel], llm: Any) -> str:
        instructions = "I'm gonna convert this raw text into valid JSON."
        if not self._is_gpt(llm):
            model_schema = PydanticSchemaParser(model=model).get_schema()
            instructions = f"{instructions}\n\nThe json should have the following structure, with the following keys:\n{model_schema}"
        return instructions

    def _save_output(self, content: str) -> None:
        if not self.output_file:
            raise Exception("Output file path is not set.")

        directory = os.path.dirname(self.output_file)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        with open(self.output_file, "w", encoding="utf-8") as file:
            file.write(content)

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
