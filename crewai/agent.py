"""Generic agent."""

from typing import Any, List, Optional

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.tools.render import render_text_description
from pydantic.v1 import BaseModel, Field, PrivateAttr, root_validator

from crewai.prompts import Prompts


class Agent(BaseModel):
    """
    Represents an agent in a system.

    Each agent has a role, a goal, a backstory, and an optional language model (llm).
    The agent can also have memory, can operate in verbose mode, and can delegate tasks to other agents.

    Attributes:
            agent_executor: An instance of the AgentExecutor class.
            role: The role of the agent.
            goal: The objective of the agent.
            backstory: The backstory of the agent.
            llm: The language model that will run the agent.
            memory: Whether the agent should have memory or not.
            verbose: Whether the agent execution should be in verbose mode.
            allow_delegation: Whether the agent is allowed to delegate tasks to other agents.
    """

    agent_executor: AgentExecutor = None
    role: str = Field(description="Role of the agent")
    goal: str = Field(description="Objective of the agent")
    backstory: str = Field(description="Backstory of the agent")
    llm: Optional[Any] = Field(description="LLM that will run the agent")
    memory: bool = Field(
        description="Whether the agent should have memory or not", default=True
    )
    verbose: bool = Field(
        description="Verbose mode for the Agent Execution", default=False
    )
    allow_delegation: bool = Field(
        description="Allow delegation of tasks to agents", default=True
    )
    tools: List[Any] = Field(description="Tools at agents disposal", default=[])
    _task_calls: List[Any] = PrivateAttr()

    @root_validator(pre=True)
    def check_llm(_cls, values):
        if not values.get("llm"):
            values["llm"] = ChatOpenAI(temperature=0.7, model_name="gpt-4")
        return values

    def __init__(self, **data):
        super().__init__(**data)
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
            prompt = Prompts.TASK_EXECUTION_WITH_MEMORY_PROMPT
        else:
            prompt = Prompts.TASK_EXECUTION_PROMPT

        execution_prompt = prompt.partial(
            goal=self.goal,
            role=self.role,
            backstory=self.backstory,
        )

        bind = self.llm.bind(stop=["\nObservation"])
        inner_agent = (
            agent_args | execution_prompt | bind | ReActSingleInputOutputParser()
        )

        self.agent_executor = AgentExecutor(agent=inner_agent, **executor_args)

    def execute_task(
        self, task: str, context: str = None, tools: List[Any] = None
    ) -> str:
        """
        Execute a task with the agent.
                Parameters:
                        task (str): Task to execute
                Returns:
                        output (str): Output of the agent
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
            }
        )["output"]

    def __tools_names(self, tools) -> str:
        return ", ".join([t.name for t in tools])
