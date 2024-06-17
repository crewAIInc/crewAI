from crewai import Agent, Process, Task, Crew, CustomAgent
from langchain.agents import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun


# # MOCKING A CUSTOM AGENT
llm = ChatOpenAI(model="gpt-4o", temperature=0)


@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)


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
search = DuckDuckGoSearchRun()
tools = [search]
llm_with_tools = llm.bind_tools(tools)

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


agent1: Agent = CustomAgent(
    agent_executor=AgentExecutor(
        agent=custom_langchain_agent, tools=tools, verbose=True
    ).invoke,
    role="Senior Research Analyst",
    goal="Discover innovative AI technologies",
    backstory="""You're a senior research analyst at a large company.
        You're responsible for analyzing data and providing insights
        to the business.
        You're currently working on a project to analyze the
        trends and innovations in the space of artificial intelligence.""",
    tools=[DuckDuckGoSearchRun()],
)

agent2 = Agent(
    role="fun fact teller",
    goal="you create a story based of the word: {input}",
    backstory="you are a wiz at telling fun facts based of the word.",
    verbose=True,
)

research_task = Task(
    description="Identify breakthrough AI technologies",
    agent=agent1,
    expected_output="A bullet list summary of the top 5 most important AI news",
)
write_article_task = Task(
    description="Draft an article on the latest AI technologies",
    agent=agent2,
    expected_output="3 paragraph blog post on the latest AI technologies",
)

my_crew = Crew(
    agents=[agent1, agent2],
    tasks=[research_task, write_article_task],
    process=Process.sequential,
    full_output=False,
)
crew = my_crew.kickoff()
print(crew)
