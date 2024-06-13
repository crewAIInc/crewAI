from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import ReActAgent

from pydantic import Field
from typing import Any

from crewai import Agent
from langchain.agents import AgentExecutor


# define sample Tool
def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b


multiply_tool = FunctionTool.from_defaults(fn=multiply)

# initialize llm
llm = OpenAI(model="gpt-3.5-turbo-0613")

# initialize ReAct agent
agent = ReActAgent.from_tools([multiply_tool], llm=llm, verbose=True)


class CustomAgentWrapper(Agent):
    custom_agent: Any = Field(default=None)
    agent_executor: Any = Field(default=None)

    def __init__(self, custom_agent, **data):
        super().__init__(**data)
        self.custom_agent = custom_agent
        self.agent_executor = AgentExecutor(agent=self.custom_agent, tools=self.tools)
        self.set_cache_handler(self.cache_handler)  # Ensure cache handler is set

    def execute_task(self, task, context=None, tools=None):
        return super().execute_task(task, context, tools)


agent1 = CustomAgentWrapper(
    custom_agent=agent,
    role="Word smith",
    goal="You are very powerful assistant",
    backstory="word smith since the you arrived",
)
print("agent1", agent1)
