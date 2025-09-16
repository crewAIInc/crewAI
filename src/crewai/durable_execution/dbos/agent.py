from typing import (
    Any,
    cast,
)

from pydantic import Field, InstanceOf, model_validator

from crewai.agent import Agent
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.llm import BaseLLM
from crewai.task import Task
from crewai.tools import BaseTool


class DBOSAgent(BaseAgent):
    """Wrap an agent for use in DBOS durable workflows.

    Automatically wraps LLM calls (both llm and function_calling_llm) as DBOS steps.

    Attributes:
            wrapped_agent: The underlying agent instance.
            llm_step_config: The DBOS step configuration to use for LLM steps.
            function_calling_llm_step_config:  The DBOS step configuration to use for function calling LLM steps.
    """

    wrapped_agent: Agent = Field(..., description="The underlying agent instance.")
    llm_step_config: dict[str, Any] | None = Field(
        default=None,
        description="The DBOS step configuration to use for LLM steps.",
    )
    function_calling_llm_step_config: dict[str, Any] | None = Field(
        default=None,
        description="The DBOS step configuration to use for function calling LLM steps.",
    )
    llm: str | InstanceOf[BaseLLM] | Any = None
    function_calling_llm: str | InstanceOf[BaseLLM] | Any | None = None

    @model_validator(mode="before")
    @classmethod
    def pre_init_setup(cls, values):
        if "wrapped_agent" not in values or not isinstance(
            values["wrapped_agent"], Agent
        ):
            raise ValueError(
                "wrapped_agent must be provided and be an instance of Agent"
            )
        # populate required fields: role, goal, backstory
        agent = cast(Agent, values["wrapped_agent"])
        values["role"] = agent.role
        values["goal"] = agent.goal
        values["backstory"] = agent.backstory
        return values

    @model_validator(mode="after")
    def post_init_setup(self):
        # TODO: wrap LLMs as steps
        self.llm = self.wrapped_agent.llm
        self.function_calling_llm = self.wrapped_agent.function_calling_llm
        return self

    def __getattr__(self, name):
        """Delegate unknown attributes to the wrapped agent."""
        if name in [
            "wrapped_agent",
            "llm_step_config",
            "function_calling_llm_step_config",
            "llm",
            "function_calling_llm",
        ]:
            return self.__getattribute__(name)
        return getattr(self.wrapped_agent, name)

    def create_agent_executor(
        self, tools: list[BaseTool] | None = None, task=None
    ) -> None:
        self.wrapped_agent.create_agent_executor(tools=tools, task=task)

    def execute_task(
        self,
        task: Task,
        context: str | None = None,
        tools: list[BaseTool] | None = None,
    ) -> str:
        return self.wrapped_agent.execute_task(
            task=task,
            context=context,
            tools=tools,
        )

    def get_delegation_tools(self, agents: list[BaseAgent]):
        return self.wrapped_agent.get_delegation_tools(agents=agents)
