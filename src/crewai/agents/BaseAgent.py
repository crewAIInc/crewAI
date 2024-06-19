from copy import deepcopy
import uuid
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod
# from langchain_openai import ChatOpenAI

from pydantic import UUID4, BaseModel, Field, InstanceOf

from crewai.utilities import I18N
from crewai.agents import CacheHandler, ToolsHandler


class BaseAgent(ABC, BaseModel):
    """Abstract base class for agents."""

    id: UUID4 = Field(default_factory=uuid.uuid4, frozen=True)
    role: str = Field(description="Role of the agent")
    goal: str = Field(description="Objective of the agent")
    backstory: str = Field(description="Backstory of the agent")
    cache: bool = Field(
        default=True, description="Whether the agent should use a cache for tool usage."
    )
    config: Optional[Dict[str, Any]] = Field(
        description="Configuration for the agent", default=None
    )
    max_rpm: Optional[int] = Field(
        default=None,
        description="Maximum number of requests per minute for the agent execution to be respected.",
    )
    verbose: bool = Field(
        default=False, description="Verbose mode for the Agent Execution"
    )
    allow_delegation: bool = Field(
        default=True, description="Allow delegation of tasks to agents"
    )
    tools: Optional[List[Any]] = Field(
        default_factory=list, description="Tools at agents' disposal"
    )
    max_iter: Optional[int] = Field(
        default=25, description="Maximum iterations for an agent to execute a task"
    )
    max_execution_time: Optional[int] = Field(
        default=None,
        description="Maximum execution time for an agent to execute a task",
    )
    agent_executor: InstanceOf = Field(
        default=None, description="An instance of the CrewAgentExecutor class."
    )
    step_callback: Optional[Any] = Field(
        default=None,
        description="Callback to be executed after each step of the agent execution.",
    )
    llm: Any = Field(
        default=None, description="Language model that will run the agent."
    )
    function_calling_llm: Optional[Any] = Field(
        description="Language model that will handle tool calling for this agent.",
        default=None,
    )
    callbacks: Optional[List[Any]] = Field(
        default=None, description="Callback to be executed"
    )
    system_template: Optional[str] = Field(
        default=None, description="System format for the agent."
    )
    prompt_template: Optional[str] = Field(
        default=None, description="Prompt format for the agent."
    )
    response_template: Optional[str] = Field(
        default=None, description="Response format for the agent."
    )
    crew: Any = Field(default=None, description="Crew to which the agent belongs.")
    i18n: I18N = Field(default=I18N(), description="Internationalization settings.")
    tools_handler: InstanceOf[ToolsHandler] = Field(
        default=None, description="An instance of the ToolsHandler class."
    )
    cache_handler: InstanceOf[CacheHandler] = Field(
        default=None, description="An instance of the CacheHandler class."
    )
    # llm: Any = Field(
    #     default_factory=lambda: ChatOpenAI(
    #         model=os.environ.get("OPENAI_MODEL_NAME", "gpt-4o")
    #     ),
    #     description="Language model that will run the agent.",
    # )

    _original_role: str | None = None
    _original_goal: str | None = None
    _original_backstory: str | None = None
    # _token_process: Optional[TokenProcess] = None
    # token_process: Any| None = None

    @abstractmethod
    def execute_task(
        self,
        task: Any,
        context: Optional[str] = None,
        tools: Optional[List[Any]] = None,
    ) -> str:
        pass

    @abstractmethod
    def set_cache_handler(self, cache_handler: Any) -> None:
        pass

    @abstractmethod
    def set_rpm_controller(self, rpm_controller: Any) -> None:
        pass

    @abstractmethod
    def create_agent_executor(self, tools=None) -> None:
        pass

    @abstractmethod
    def _parse_tools(self, tools: List[Any]) -> List[Any]:
        pass

    # @abstractmethod
    def interpolate_inputs(self, inputs: Dict[str, Any]) -> None:
        """Interpolate inputs into the agent description and backstory."""
        if self._original_role is None:
            self._original_role = self.role
        if self._original_goal is None:
            self._original_goal = self.goal
        if self._original_backstory is None:
            self._original_backstory = self.backstory

        if inputs:
            self.role = self._original_role.format(**inputs)
            self.goal = self._original_goal.format(**inputs)
            self.backstory = self._original_backstory.format(**inputs)
        pass

    def increment_formatting_errors(self) -> None:
        print("Formatting errors incremented")

    def format_log_to_str(
        self,
        intermediate_steps: List[Any],
        observation_prefix: str = "Observation: ",
        llm_prefix: str = "",
    ) -> str:
        return "Formatted log"

    def copy(self):
        exclude = {
            "id",
            "_logger",
            "_rpm_controller",
            "_request_within_rpm_limit",
            "_token_process",
            "agent_executor",
            "tools",
            "tools_handler",
            "cache_handler",
        }

        copied_data = self.model_dump(exclude=exclude)
        copied_data = {k: v for k, v in copied_data.items() if v is not None}

        copied_agent = self(**copied_data)
        copied_agent.tools = deepcopy(self.tools)

        return copied_agent


# Implementing the original Agent class as its own standalone class
# class Agent(BaseAgent):
#     """Represents an agent in a system."""

#     __hash__ = object.__hash__  # type: ignore
#     _logger: Any = PrivateAttr()
#     _rpm_controller: Any = PrivateAttr(default=None)
#     _request_within_rpm_limit: Any = PrivateAttr(default=None)
#     _token_process: Any = PrivateAttr()

#     def __init__(self, **data):
#         config = data.pop("config", {})
#         super().__init__(**config, **data)
#         # self._original_role = None
#         # self._original_goal = None
#         # self._original_backstory = None
#         # self.formatting_errors = 0

#     def execute_task(self, task: Any, context: Optional[str] = None, tools: Optional[List[Any]] = None) -> str:
#         # Implement the task execution logic here
#         print(f"Executing task: {task} with context: {context}")
#         return "Task executed"

#     def set_cache_handler(self, cache_handler: Any) -> None:
#         # Implement the cache handler setup logic here
#         print("Cache handler set")

#     def set_rpm_controller(self, rpm_controller: Any) -> None:
#         # Implement the RPM controller setup logic here
#         print("RPM controller set")

#     def create_agent_executor(self, tools=None) -> None:
#         # Implement the agent executor creation logic here
#         print("Agent executor created")

#     def interpolate_inputs(self, inputs: Dict[str, Any]) -> None:
#         # Implement the input interpolation logic here
#         print("Inputs interpolated")

#     def increment_formatting_errors(self) -> None:
#         # Implement the logic to increment formatting errors here
#         print("Formatting errors incremented")

#     def format_log_to_str(self, intermediate_steps: List[Any], observation_prefix: str = "Observation: ", llm_prefix: str = "") -> str:
#         # Implement the logic to format the log to a string here
#         return "Formatted log"

#     def copy(self):
#         """Create a deep copy of the Agent."""
#         exclude = {
#             "id",
#             "_logger",
#             "_rpm_controller",
#             "_request_within_rpm_limit",
#             "_token_process",
#             "agent_executor",
#             "tools",
#             "tools_handler",
#             "cache_handler",
#         }

#         copied_data = self.dict(exclude=exclude)
#         copied_data = {k: v for k, v in copied_data.items() if v is not None}

#         copied_agent = Agent(**copied_data)
#         copied_agent.tools = deepcopy(self.tools)

#         return copied_agent

#     def _parse_tools(self, tools: List[Any]) -> List[Any]:
#         """Parse tools to be used for the task."""
#         # Implement the logic to parse tools here
#         return tools

#     @staticmethod
#     def __tools_names(tools) -> str:
#         return ", ".join([t.name for t in tools])

#     def __repr__(self):
#         return f"Agent(role={self.role}, goal={self.goal}, backstory={self.backstory})"

# # Example usage
# if __name__ == "__main__":
#     agent = Agent(
#         role="SampleRole",
#         goal="SampleGoal",
#         backstory="SampleBackstory",
#         llm="SampleLLM"
#     )
#     task = {"type": "sample_task", "data": "some data"}
#     result = agent.execute_task(task)
#     print(result)


# abstract away all Agent

# set_cache_handler: llama index version
# set_cache_handler: auto gens version


## execute_task = Agent_executor for each


# def set_agent_executor():
# pass

# AutoGenAgent(BaseAgent):
#   def nset_agent_executor(content) -> str:

#   autogen.chat(messages=[{"content": content}])

#   str


#  agent_executor: An instance of the CrewAgentExecutor class.
#             role: The role of the agent.
#             goal: The objective of the agent.
#             backstory: The backstory of the agent.
#             config: Dict representation of agent configuration.
#             llm: The language model that will run the agent.
#             function_calling_llm: The language model that will handle the tool calling for this agent, it overrides the crew function_calling_llm.
#             max_iter: Maximum number of iterations for an agent to execute a task.
#             memory: Whether the agent should have memory or not.
#             max_rpm: Maximum number of requests per minute for the agent execution to be respected.
#             verbose: Whether the agent execution should be in verbose mode.
#             allow_delegation: Whether the agent is allowed to delegate tasks to other agents.
#             tools: Tools at agents disposal
#             step_callback: Callback to be executed after each step of the agent execution.
#             callbacks: A list of callback functions from the langchain library that are triggered during the agent's execution process


# we have our own base requirements to RUN - what are requirements for the agent to run?

# execute_task = successfully -> str
# llama

# agent1 = Agent(
#   role=''
#   backstory=''
# )

# agent2 = AutoGenAgent()

# mvp end goal base requirements for ANY agent
# -> execute_task

# do we need goal/backstory  max_rpm?
