import uuid
from typing import Any, Dict, List, Optional, Protocol
from pydantic import UUID4, BaseModel, Field
from copy import deepcopy


class BaseAgentProtocol(Protocol):
    id: UUID4
    role: str
    goal: str
    backstory: str
    cache: bool
    config: Optional[Dict[str, Any]]
    max_rpm: Optional[int]
    verbose: bool
    allow_delegation: bool
    tools: Optional[List[Any]]
    max_iter: Optional[int]
    max_execution_time: Optional[int]
    agent_executor: Any
    step_callback: Optional[Any]
    llm: Any
    function_calling_llm: Optional[Any]
    callbacks: Optional[List[Any]]
    system_template: Optional[str]
    prompt_template: Optional[str]
    response_template: Optional[str]
    crew: Any
    i18n: Any
    tools_handler: Any
    cache_handler: Any

    def execute_task(
        self,
        task: Any,
        context: Optional[str] = None,
        tools: Optional[List[Any]] = None,
    ) -> str: ...

    def set_cache_handler(self, cache_handler: Any) -> None: ...

    def set_rpm_controller(self, rpm_controller: Any) -> None: ...

    def create_agent_executor(self, tools=None) -> None: ...

    def interpolate_inputs(self, inputs: Dict[str, Any]) -> None: ...

    def increment_formatting_errors(self) -> None: ...

    def format_log_to_str(
        self,
        intermediate_steps: List[Any],
        observation_prefix: str = "Observation: ",
        llm_prefix: str = "",
    ) -> str: ...

    def copy(self) -> "BaseAgentProtocol": ...

    def _parse_tools(self, tools: List[Any]) -> List[Any]: ...


class BaseAgent(BaseModel):
    """Concrete implementation of the BaseAgentProtocol."""

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
    agent_executor: Any = Field(
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
    i18n: Any = Field(default=None, description="Internationalization settings.")
    tools_handler: Any = Field(
        default=None, description="An instance of the ToolsHandler class."
    )
    cache_handler: Any = Field(
        default=None, description="An instance of the CacheHandler class."
    )

    _original_role: Optional[str] = None
    _original_goal: Optional[str] = None
    _original_backstory: Optional[str] = None

    def execute_task(
        self,
        task: Any,
        context: Optional[str] = None,
        tools: Optional[List[Any]] = None,
    ) -> str:
        raise NotImplementedError

    def set_cache_handler(self, cache_handler: Any) -> None:
        raise NotImplementedError

    def set_rpm_controller(self, rpm_controller: Any) -> None:
        raise NotImplementedError

    def create_agent_executor(self, tools=None) -> None:
        raise NotImplementedError

    def interpolate_inputs(self, inputs: Dict[str, Any]) -> None:
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

    def increment_formatting_errors(self) -> None:
        print("Formatting errors incremented")

    def format_log_to_str(
        self,
        intermediate_steps: List[Any],
        observation_prefix: str = "Observation: ",
        llm_prefix: str = "",
    ) -> str:
        return "Formatted log"

    def copy(self) -> "BaseAgent":
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

        copied_agent = self.__class__(**copied_data)
        copied_agent.tools = deepcopy(self.tools)

        return copied_agent

    def _parse_tools(self, tools: List[Any]) -> List[Any]:
        raise NotImplementedError
