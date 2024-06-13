from crewai import Task, Crew, Process, Agent
# from crewai.agents import CustomAgentWrapper

# from crewai.agent import Agent

from langchain.agents import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from pydantic import Field
from typing import Any
from pprint import pprint
from crewai.utilities import I18N


# MOCKING A CUSTOM AGENT
llm = ChatOpenAI(model="gpt-4o", temperature=0)


@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)


get_word_length.invoke("abc")


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are very powerful assistant, but don't know current events",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
tools = [get_word_length]
llm_with_tools = llm.bind_tools(tools)


class CustomAgentWrapper(Agent):
    custom_agent: Any = Field(default=None)
    agent_executor: Any = Field(default=None)
    i18n: I18N = Field(default=I18N())

    def __init__(self, custom_agent, agent_executor, **data):
        super().__init__(**data)
        self.custom_agent = custom_agent
        self.agent_executor = agent_executor

    def execute_task(self, task, context=None, tools=None):
        return super().execute_task(task, context, tools)

    # def i18n(self, value: I18N) -> None:
    #     if hasattr(self, "_agent") and hasattr(self._agent, "i18n"):
    #         self._agent.i18n = value
    #     else:
    #         self._i18n = value


custom_langchain_agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)
print("custom_langchain_agent", custom_langchain_agent)

agent1: Agent = CustomAgentWrapper(
    custom_agent=custom_langchain_agent,
    agent_executor=AgentExecutor(agent=custom_langchain_agent, tools=tools),
    role="Word smith",
    goal="You are very powerful assistant",
    backstory="word smith since the you arrived",
    verbose=True,
)

agent2 = Agent(
    role="fun fact teller",
    goal="you create a story based of the word given",
    backstory="you are a wiz at telling fun facts based of the word and the number of letter is has.",
    verbose=True,
)
print("AGENT 1")
pprint(vars(agent1))
pprint(vars(agent1))
print(
    "\n AGENT 2",
)
pprint(vars(agent2))


task1 = Task(
    agent=agent1,
    description="how many letters in {input}",
    expected_output="give me the word and the number of letter of words.",
)

task2 = Task(
    agent=agent2,
    description="create a paragraph about a fun fact for the {input} and the significance of the number of letter of words.",
    expected_output="paragraph about a fun fact for the given word and the significance of the number of letter of words.",
)

my_crew = Crew(
    agents=[agent1, agent2],
    tasks=[task1, task2],
    process=Process.sequential,
    cache=True,
)

crew = my_crew.kickoff(inputs={"input": "grapes"})
print("RESULT:", crew)


# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# list(agent_executor.stream({"input": "How many letters in the word eudca"}))


# should convert to for crewai

# '''
# ThirdPartyAgent(

# )
# '''
