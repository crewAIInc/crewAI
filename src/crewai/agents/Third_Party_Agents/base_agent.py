from copy import deepcopy
import uuid
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod
from pydantic import UUID4, BaseModel, Field, InstanceOf, model_validator

from crewai.utilities import I18N, RPMController, Logger
from crewai.agents import CacheHandler, ToolsHandler


class BaseAgent(ABC, BaseModel):
    """Abstract Base Class for for all third party agents."""

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

    _original_role: str | None = None
    _original_goal: str | None = None
    _original_backstory: str | None = None

    @abstractmethod
    def execute_task(
        self,
        task: Any,
        context: Optional[str] = None,
        tools: Optional[List[Any]] = None,
    ) -> str:
        pass

    @abstractmethod
    def create_agent_executor(self, tools=None) -> None:
        pass

    @abstractmethod
    def _parse_tools(self, tools: List[Any]) -> List[Any]:
        pass

    @abstractmethod
    def create_delegate_work_tool(self, agents):
        """Extend this method to create a delegate work tool for the agent."""
        pass

    @abstractmethod
    def create_ask_question_tool(self, agents):
        """Extend this method to create an ask question tool for the agent."""
        pass

    @abstractmethod
    def set_agent_tools(self, agents: List["BaseAgent"]):
        """Set the agent tools that init BaseAgenTools class."""
        pass

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

    def set_cache_handler(self, cache_handler: CacheHandler) -> None:
        """Set the cache handler for the agent.

        Args:
            cache_handler: An instance of the CacheHandler class.
        """
        self.tools_handler = ToolsHandler()
        if self.cache:
            self.cache_handler = cache_handler
            self.tools_handler.cache = cache_handler
        self.create_agent_executor()

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

    def set_rpm_controller(self, rpm_controller: RPMController) -> None:
        """Set the rpm controller for the agent.

        Args:
            rpm_controller: An instance of the RPMController class.
        """
        if not self._rpm_controller:
            self._rpm_controller = rpm_controller
            self.create_agent_executor()

    @model_validator(mode="after")
    def set_private_attrs(self):
        """Set private attributes."""
        self._logger = Logger(self.verbose)
        if self.max_rpm and not self._rpm_controller:
            self._rpm_controller = RPMController(
                max_rpm=self.max_rpm, logger=self._logger
            )
        return self
