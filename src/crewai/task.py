from copy import deepcopy
import os
import re
import threading
import uuid
from typing import Any, Dict, List, Optional, Type

from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from langchain_openai import ChatOpenAI
from pydantic import UUID4, BaseModel, Field, field_validator, model_validator
from pydantic_core import PydanticCustomError

from crewai.agent import Agent
from crewai.tasks.task_output import TaskOutput
from crewai.utilities import I18N, Converter, ConverterError, Printer
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
    thread: Optional[threading.Thread] = None
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
    agent: Optional[Agent] = Field(
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
    rci: Optional[bool] = Field(
        default=True,
        description="Whether the agent should use Recursive Criticism and Iteration(RCI) to verify output or not",
    )
    rci_depth: Optional[int] = Field(
        default=1,
        description="If the agent uses RCI, how many iterations it can maximum re-iterate.",
    )

    _original_description: str | None = None
    _original_expected_output: str | None = None

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

    def execute(  # type: ignore # Missing return statement
        self,
        agent: Agent | None = None,
        context: Optional[str] = None,
        tools: Optional[List[Any]] = None,
    ) -> str:
        """Execute the task.

        Returns:
            Output of the task.
        """

        agent = agent or self.agent
        if not agent:
            raise Exception(
                f"The task '{self.description}' has no agent assigned, therefore it can't be executed directly and should be executed in a Crew using a specific process that support that, like hierarchical."
            )

        if self.context:
            # type: ignore # Incompatible types in assignment (expression has type "list[Never]", variable has type "str | None")
            context = []
            for task in self.context:
                if task.async_execution:
                    task.thread.join()  # type: ignore # Item "None" of "Thread | None" has no attribute "join"
                if task and task.output:
                    # type: ignore # Item "str" of "str | None" has no attribute "append"
                    context.append(task.output.raw_output)
            # type: ignore # Argument 1 to "join" of "str" has incompatible type "str | None"; expected "Iterable[str]"
            context = "\n".join(context)

        self.prompt_context = context
        tools = tools or self.tools

        # To check if the rci_depth is 0 and rci is set True, if True then set rci_depth as 1
        if self.rci and self.rci_depth == 0:
            self.rci_depth = 1

        if self.async_execution:
            self.thread = threading.Thread(
                target=self._execute, args=(agent, self, context, tools)
            )
            self.thread.start()
        else:
            result = self._execute(
                task=self,
                agent=agent,
                context=context,
                tools=tools,
                rci = self.rci, # adding rci in function call
                rci_depth=self.rci_depth
            )
            return result

    # Create new methods that will verify the output of the agent using RCI - Recursive Criticism and Iteration
    def critique(self, agent, task, output, llm):

        critic_prompt_template = """
        {agent_backstory}. + You are a great critic who has keen eyes for errors. 
        A task is assigned to a LLM model which provides an output based on the task description. 
        Identify errors in the output provided by the model, if any. 
        Strictly avoid grammatical rephrasing or paraphrasing, point out only the logical or factual inaccuracies. 
        Do not reproduce your own version of the output, just provide only the critique.
        I repeat do not give your own rewritten version of the output. Only point out the errors.

        Provided Task: {task_description}
        Output: {output}
        """

        critic_prompt = PromptTemplate(
            input_variables=["task_description", "result"],
            template=critic_prompt_template
        )

        critic_chain = critic_prompt | llm | StrOutputParser()
        
        critic_response = critic_chain.invoke({
            "agent_backstory": agent.backstory,
            "task_description": task.description,
            "output": output
        })

        print("Critic Response:\n", critic_response)

        return critic_response

    
    def validate(self, task, critique, output, llm):

        validate_prompt_template = """
        You are a context analyzer and your job is to identify if the critique to a task provided and its corresponding output, states "significant changes are required in the output" or synonymous phrases of that significance order.
        If there are significant changes stated in the critique, without any preamble or additional explanation, just print "True" else just print "False". 
        Avoid minor inaccuracies or minor adjustments. 
        If the critique says the overall output matches the task description with minor changes required, print 'False'.
        Unless and until the output change provided by the critique is tangential, print 'False', if it is tagential print 'True'. 
        Make sure the output provided by you should be only one word that is either 'True' or 'False'.

        Provided Task: {task_description}
        Output: {output}
        Critique: {critique}
        """

        validate_prompt = PromptTemplate(
            input_variables=["task_description", "output", "critique"],
            template=validate_prompt_template
        )

        validate_chain = validate_prompt | llm | StrOutputParser()

        validate_response = validate_chain.invoke({
            "task_description": task.description,
            "output": output,
            "critique": critique
        })

        print("Validate Response:\n", validate_response)
        return validate_response
    
    def improve(self, task, output, critique, llm):
    
        improve_prompt_template = """
        You are a helpful assistant who can skillfully analyze the critique provided to a task based on its output. 
        Your job is completely understand the task, its corresponding output and the critique provided and rewrite the output based on the critique. 
        Avoid writing "Here is my modificication" or any synoymous phrases at the start or at the end.
        No need for premable or explanations from your side, just rewrite the output based on the critique. 
        Make sure you maintain the format if mentioned in the task description.

        Provided Task: {task_description}
        Output: {output}
        Critique: {critique}
        """

        improve_prompt = PromptTemplate(
            input_variables=["task_description", "output", "critique"],
            template=improve_prompt_template
        )

        improve_chain = improve_prompt | llm | StrOutputParser()
        
        improve_response = improve_chain.invoke({
            "task_description": task.description,
            "output": output,
            "critique": critique
        })

        print("Improvised Response:\n", improve_response)

        return improve_response

    def _execute(self, agent, task, context, tools, rci=True, rci_depth=1):
        result = agent.execute_task(
            task=task,
            context=context,
            tools=tools,
        )

        # To perform RCI if rci is set to True
        llm = ChatOllama(model="llama3")
        depth = 0
        while rci and (depth < rci_depth):
            critic_response = self.critique(llm = llm, agent=agent, task=task, output=result)
            validate_response = self.validate(llm=llm, task=task, output=result, critique=critic_response)
            if validate_response == "True": 
                result = self.improve(llm=llm, task=task, output=result,critique=critic_response)
            else:
                rci = False
            
            depth += 1

        exported_output = self._export_output(result)

        self.output = TaskOutput(
            description=self.description,
            exported_output=exported_output,
            raw_output=result,
            agent=agent.role,
        )

        if self.callback:
            self.callback(self.output)

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

    def copy(self):
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
        cloned_agent = self.agent.copy() if self.agent else None
        cloned_tools = deepcopy(self.tools) if self.tools else None

        copied_task = Task(
            **copied_data,
            context=cloned_context,
            agent=cloned_agent,
            tools=cloned_tools,
        )
        return copied_task

    def _export_output(self, result: str) -> Any:
        exported_result = result
        instructions = "I'm gonna convert this raw text into valid JSON."

        if self.output_pydantic or self.output_json:
            model = self.output_pydantic or self.output_json

            # try to convert task_output directly to pydantic/json
            try:
                # type: ignore # Item "None" of "type[BaseModel] | None" has no attribute "model_validate_json"
                exported_result = model.model_validate_json(result)
                if self.output_json:
                    # type: ignore # "str" has no attribute "model_dump"
                    return exported_result.model_dump()
                return exported_result
            except Exception:
                # sometimes the response contains valid JSON in the middle of text
                match = re.search(r"({.*})", result, re.DOTALL)
                if match:
                    try:
                        # type: ignore # Item "None" of "type[BaseModel] | None" has no attribute "model_validate_json"
                        exported_result = model.model_validate_json(match.group(0))
                        if self.output_json:
                            # type: ignore # "str" has no attribute "model_dump"
                            return exported_result.model_dump()
                        return exported_result
                    except Exception:
                        pass

            # type: ignore # Item "None" of "Agent | None" has no attribute "function_calling_llm"
            llm = self.agent.function_calling_llm or self.agent.llm

            if not self._is_gpt(llm):
                # type: ignore # Argument "model" to "PydanticSchemaParser" has incompatible type "type[BaseModel] | None"; expected "type[BaseModel]"
                model_schema = PydanticSchemaParser(model=model).get_schema()
                instructions = f"{instructions}\n\nThe json should have the following structure, with the following keys:\n{model_schema}"

            converter = Converter(
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
                # type: ignore # "str" has no attribute "json"
                exported_result if not self.output_pydantic else exported_result.json()
            )
            self._save_file(content)

        return exported_result

    def _is_gpt(self, llm) -> bool:
        return isinstance(llm, ChatOpenAI) and llm.openai_api_base is None

    def _save_file(self, result: Any) -> None:
        # type: ignore # Value of type variable "AnyOrLiteralStr" of "dirname" cannot be "str | None"
        directory = os.path.dirname(self.output_file)

        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # type: ignore # Argument 1 to "open" has incompatible type "str | None"; expected "int | str | bytes | PathLike[str] | PathLike[bytes]"
        with open(self.output_file, "w", encoding="utf-8") as file:
            file.write(result)
        return None

    def __repr__(self):
        return f"Task(description={self.description}, expected_output={self.expected_output})"
