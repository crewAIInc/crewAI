from crewai import Agent
# from crewai.agents import CustomAgentWrapper

# from crewai.agent import Agent

from langchain.agents import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_openai import ChatOpenAI

# MOCKING A CUSTOM AGENT
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


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


agent = (
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

agent1 = Agent(
    custom_agent=agent,
    role="Word smith",
    goal="You are very powerful assistant",
    backstory="word smith since the you arrived",
)
print("agent1", agent1)
