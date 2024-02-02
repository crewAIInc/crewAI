import uuid
from typing import Any, List, Optional

from langchain.agents.agent import RunnableAgent
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.memory import ConversationSummaryMemory
from langchain.tools.render import render_text_description
from langchain_core.runnables.config import RunnableConfig
from langchain_openai import ChatOpenAI
from pydantic import (
    UUID4,
    BaseModel,
    ConfigDict,
    Field,
    InstanceOf,
    PrivateAttr,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from crewai.agents import (
    CacheHandler,
    CrewAgentExecutor,
    CrewAgentOutputParser,
    ToolsHandler,
)
from crewai.utilities import I18N, Logger, Prompts, RPMController


class Agent(BaseModel):
    """Represents an agent in a system.

    Each agent has a role, a goal, a backstory, and an optional language model (llm).
    The agent can also have memory, can operate in verbose mode, and can delegate tasks to other agents.

    Attributes:
            agent_executor: An instance of the CrewAgentExecutor class.
            role: The role of the agent.
            goal: The objective of the agent.
            backstory: The backstory of the agent.
            llm: The language model that will run the agent.
            max_iter: Maximum number of iterations for an agent to execute a task.
            memory: Whether the agent should have memory or not.
            max_rpm: Maximum number of requests per minute for the agent execution to be respected.
            verbose: Whether the agent execution should be in verbose mode.
            allow_delegation: Whether the agent is allowed to delegate tasks to other agents.
            tools: Tools at agents disposal
    """

    __hash__ = object.__hash__  # type: ignore
    _logger: Logger = PrivateAttr()
    _rpm_controller: RPMController = PrivateAttr(default=None)
    _request_within_rpm_limit: Any = PrivateAttr(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)
    id: UUID4 = Field(
        default_factory=uuid.uuid4,
        frozen=True,
        description="Unique identifier for the object, not set by user.",
    )
    role: str = Field(description="Role of the agent")
    goal: str = Field(description="Objective of the agent")
    backstory: str = Field(description="Backstory of the agent")
    max_rpm: Optional[int] = Field(
        default=None,
        description="Maximum number of requests per minute for the agent execution to be respected.",
    )
    memory: bool = Field(
        default=True, description="Whether the agent should have memory or not"
    )
    verbose: bool = Field(
        default=False, description="Verbose mode for the Agent Execution"
    )
    allow_delegation: bool = Field(
        default=True, description="Allow delegation of tasks to agents"
    )
    tools: List[Any] = Field(
        default_factory=list, description="Tools at agents disposal"
    )
    max_iter: Optional[int] = Field(
        default=15, description="Maximum iterations for an agent to execute a task"
    )
    agent_executor: InstanceOf[CrewAgentExecutor] = Field(
        default=None, description="An instance of the CrewAgentExecutor class."
    )
    tools_handler: InstanceOf[ToolsHandler] = Field(
        default=None, description="An instance of the ToolsHandler class."
    )
    cache_handler: InstanceOf[CacheHandler] = Field(
        default=CacheHandler(), description="An instance of the CacheHandler class."
    )
    i18n: I18N = Field(default=I18N(), description="Internationalization settings.")
    llm: Any = Field(
        default_factory=lambda: ChatOpenAI(
            model="gpt-4",
        ),
        description="Language model that will run the agent.",
    )

    @field_validator("id", mode="before")
    @classmethod
    def _deny_user_set_id(cls, v: Optional[UUID4]) -> None:
        if v:
            raise PydanticCustomError(
                "may_not_set_field", "This field is not to be set by the user.", {}
            )

    @model_validator(mode="after")
    def set_private_attrs(self):
        """Set private attributes."""
        self._logger = Logger(self.verbose)
        if self.max_rpm and not self._rpm_controller:
            self._rpm_controller = RPMController(
                max_rpm=self.max_rpm, logger=self._logger
            )
        return self

    @model_validator(mode="after")
    def check_agent_executor(self) -> "Agent":
        """Check if the agent executor is set."""
        if not self.agent_executor:
            self.set_cache_handler(self.cache_handler)
        return self

    def execute_task(
        self,
        task: str,
        context: Optional[str] = None,
        tools: Optional[List[Any]] = None,
    ) -> str:
        """Execute a task with the agent.

        Args:
            task: Task to execute.
            context: Context to execute the task in.
            tools: Tools to use for the task.

        Returns:
            Output of the agent
        """

        if context:
            task = self.i18n.slice("task_with_context").format(
                task=task, context=context
            )

        tools = tools or self.tools
        self.agent_executor.tools = tools

        result = self.agent_executor.invoke(
            {
                "input": task,
                "tool_names": self.__tools_names(tools),
                "tools": render_text_description(tools),
            },
            RunnableConfig(callbacks=[self.tools_handler]),
        )["output"]

        if self.max_rpm:
            self._rpm_controller.stop_rpm_counter()

        return result

    def set_cache_handler(self, cache_handler: CacheHandler) -> None:
        """Set the cache handler for the agent.

        Args:
            cache_handler: An instance of the CacheHandler class.
        """
        self.cache_handler = cache_handler
        self.tools_handler = ToolsHandler(cache=self.cache_handler)
        self._create_agent_executor()

    def set_rpm_controller(self, rpm_controller: RPMController) -> None:
        """Set the rpm controller for the agent.

        Args:
            rpm_controller: An instance of the RPMController class.
        """
        if not self._rpm_controller:
            self._rpm_controller = rpm_controller
            self._create_agent_executor()

    def _create_agent_executor(self) -> None:
        """Create an agent executor for the agent.

        Returns:
            An instance of the CrewAgentExecutor class.
        """
        agent_args = {
            "input": lambda x: x["input"],
            "tools": lambda x: x["tools"],
            "tool_names": lambda x: x["tool_names"],
            "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
        }
        executor_args = {
            "i18n": self.i18n,
            "tools": self.tools,
            "verbose": self.verbose,
            "handle_parsing_errors": True,
            "max_iterations": self.max_iter,
        }

        if self._rpm_controller:
            executor_args["request_within_rpm_limit"] = (
                self._rpm_controller.check_or_wait
            )

        if self.memory:
            summary_memory = ConversationSummaryMemory(
                llm=self.llm, input_key="input", memory_key="chat_history"
            )
            executor_args["memory"] = summary_memory
            agent_args["chat_history"] = lambda x: x["chat_history"]
            prompt = Prompts(i18n=self.i18n).task_execution_with_memory()
        else:
            prompt = Prompts(i18n=self.i18n).task_execution()

        execution_prompt = prompt.partial(
            goal=self.goal,
            role=self.role,
            backstory=self.backstory,
        )

        bind = self.llm.bind(stop=[self.i18n.slice("observation")])
        inner_agent = (
            agent_args
            | execution_prompt
            | bind
            | CrewAgentOutputParser(
                tools_handler=self.tools_handler,
                cache=self.cache_handler,
                i18n=self.i18n,
            )
        )
        self.agent_executor = CrewAgentExecutor(
            agent=RunnableAgent(runnable=inner_agent), **executor_args
        )

    @staticmethod
    def __tools_names(tools) -> str:
        return ", ".join([t.name for t in tools])
