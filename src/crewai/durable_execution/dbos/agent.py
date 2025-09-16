from typing import (
    Any,
    cast,
)

from pydantic import Field, PrivateAttr, model_validator

from crewai.agent import Agent
from crewai.agents.agent_builder.base_agent import BaseAgent
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

    orig_agent: Agent = Field(exclude=True, description="The original agent instance.")
    llm_step_config: dict[str, Any] | None = Field(
        default=None,
        description="The DBOS step configuration to use for LLM steps.",
    )
    function_calling_llm_step_config: dict[str, Any] | None = Field(
        default=None,
        description="The DBOS step configuration to use for function calling LLM steps.",
    )
    _wrapped_agent: Agent = PrivateAttr()
    # Field(
    #     default=None, description="A deep copy of the original agent instance."
    # )

    @model_validator(mode="before")
    @classmethod
    def pre_init_setup(cls, values):
        if "orig_agent" not in values or not isinstance(values["orig_agent"], Agent):
            raise ValueError("orig_agent must be provided and be an instance of Agent")
        # populate required fields: role, goal, backstory
        agent = cast(Agent, values["orig_agent"])
        values["role"] = agent.role
        values["goal"] = agent.goal
        values["backstory"] = agent.backstory
        return values

    @model_validator(mode="after")
    def post_init_setup(self):
        # Create a deep copy of the agent to avoid mutating the original
        self._wrapped_agent = self.orig_agent.copy()
        # Replace the original create_agent_executor with a DBOS wrap
        object.__setattr__(
            self._wrapped_agent, "create_agent_executor", self.create_agent_executor
        )
        # TODO: wrap LLM/function calling LLM as steps
        return self

    def execute_task(
        self,
        task: Task,
        context: str | None = None,
        tools: list[BaseTool] | None = None,
    ) -> str:
        return self._wrapped_agent.execute_task(
            task=task,
            context=context,
            tools=tools,
        )

    def get_delegation_tools(self, agents: list[BaseAgent]):
        return self._wrapped_agent.get_delegation_tools(agents=agents)

    def create_agent_executor(
        self, tools: list[BaseTool] | None = None, task=None
    ) -> None:
        # TODO: wrap with DBOS.
        print("DBOS create_agent_executor called")
        return self.orig_agent.create_agent_executor(tools=tools, task=task)
