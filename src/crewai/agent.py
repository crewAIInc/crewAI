import uuid
from typing import Any, List, Optional

from langchain.agents.format_scratchpad import format_log_to_str
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.tools.render import render_text_description
from langchain_core.runnables.config import RunnableConfig
from pydantic import (
    UUID4,
    BaseModel,
    ConfigDict,
    Field,
    InstanceOf,
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
from crewai.prompts import Prompts


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
            memory: Whether the agent should have memory or not.
            verbose: Whether the agent execution should be in verbose mode.
            allow_delegation: Whether the agent is allowed to delegate tasks to other agents.
    """

    __hash__ = object.__hash__
    model_config = ConfigDict(arbitrary_types_allowed=True)
    id: UUID4 = Field(
        default_factory=uuid.uuid4,
        frozen=True,
        description="Unique identifier for the object, not set by user.",
    )
    role: str = Field(description="Role of the agent")
    goal: str = Field(description="Objective of the agent")
    backstory: str = Field(description="Backstory of the agent")
    llm: Optional[Any] = Field(
        default_factory=lambda: ChatOpenAI(
            temperature=0.7,
            model_name="gpt-4",
        ),
        description="Language model that will run the agent.",
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
    agent_executor: Optional[InstanceOf[CrewAgentExecutor]] = Field(
        default=None, description="An instance of the CrewAgentExecutor class."
    )
    tools_handler: Optional[InstanceOf[ToolsHandler]] = Field(
        default=None, description="An instance of the ToolsHandler class."
    )
    cache_handler: Optional[InstanceOf[CacheHandler]] = Field(
        default=CacheHandler(), description="An instance of the CacheHandler class."
    )

    @field_validator("id", mode="before")
    @classmethod
    def _deny_user_set_id(cls, v: Optional[UUID4]) -> None:
        if v:
            raise PydanticCustomError(
                "may_not_set_field", "This field is not to be set by the user.", {}
            )

    @model_validator(mode="after")
    def check_agent_executor(self) -> "Agent":
        if not self.agent_executor:
            self.set_cache_handler(self.cache_handler)
        return self

    def execute_task(
        self, task: str, context: str = None, tools: List[Any] = None
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
            task = "\n".join(
                [task, "\nThis is the context you are working with:", context]
            )

        tools = tools or self.tools
        self.agent_executor.tools = tools

        return self.agent_executor.invoke(
            {
                "input": task,
                "tool_names": self.__tools_names(tools),
                "tools": render_text_description(tools),
            },
            RunnableConfig(callbacks=[self.tools_handler]),
        )["output"]

    def set_cache_handler(self, cache_handler) -> None:
        self.cache_handler = cache_handler
        self.tools_handler = ToolsHandler(cache=self.cache_handler)
        self.__create_agent_executor()

    def __create_agent_executor(self) -> CrewAgentExecutor:
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
            "tools": self.tools,
            "verbose": self.verbose,
            "handle_parsing_errors": True,
        }

        if self.memory:
            summary_memory = ConversationSummaryMemory(
                llm=self.llm, memory_key="chat_history", input_key="input"
            )
            executor_args["memory"] = summary_memory
            agent_args["chat_history"] = lambda x: x["chat_history"]
            prompt = Prompts().task_execution_with_memory()
        else:
            prompt = Prompts().task_execution()

        execution_prompt = prompt.partial(
            goal=self.goal,
            role=self.role,
            backstory=self.backstory,
        )

        bind = self.llm.bind(stop=["\nObservation"])
        inner_agent = (
            agent_args
            | execution_prompt
            | bind
            | CrewAgentOutputParser(
                tools_handler=self.tools_handler, cache=self.cache_handler
            )
        )
        self.agent_executor = CrewAgentExecutor(agent=inner_agent, **executor_args)

    @staticmethod
    def __tools_names(tools) -> str:
        return ", ".join([t.name for t in tools])
