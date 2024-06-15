from crewai import Agent, Process, Task, Crew, CustomAgentWrapper
from langchain.agents import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from crewai.utilities.printer import Printer


# # MOCKING A CUSTOM AGENT
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
# print("custom_langchain_agent", custom_langchain_agent)

agent1: Agent = CustomAgentWrapper(
    custom_agent=custom_langchain_agent,
    agent_executor=AgentExecutor(agent=custom_langchain_agent, tools=tools).invoke,
    role="Word smith",
    goal="You are very powerful assistant, you take a word: {input} and return a fun fact based off its letter count",
    backstory="word smith since the you arrived",
    verbose=True,
)

agent2 = Agent(
    role="fun fact teller",
    goal="you create a story based of the word: {input}",
    backstory="you are a wiz at telling fun facts based of the word and the number of letter is has.",
    verbose=True,
)
printer = Printer()
# print("AGENT 1")
# printer.print(vars(agent1), color='bold_green')
# pprint(vars(agent1))
# print(
#     "\n AGENT 2",
# )
# pprint(vars(agent2))
# printer.print(vars(agent2), color='bold_purple')

task1 = Task(
    agent=agent1,
    description="return a fun fact about {input}",
    expected_output="give me the word and the number of letter of words.",
)

# print(
#     "\n task 1",
# )

# pprint(vars(agent2))
# printer.print(vars(task1), color='bold_green')
task2 = Task(
    agent=agent2,
    description="create a paragraph about a fun fact for the given word and the significance of the number of letter of words.",
    expected_output="paragraph about a fun fact for the given word and the significance of the number of letter of words.",
    context=[task1],
)
# print(
#     "\n task 2",
# )
# printer.print(vars(task2), color='bold_green')

my_crew = Crew(
    agents=[agent1, agent2],
    tasks=[task1, task2],
    process=Process.sequential,
    cache=True,
)

crew = my_crew.kickoff(inputs={"input": "grapes"})
# print("RESULT:", crew)


# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# list(agent_executor.stream({"input": "How many letters in the word eudca"}))


# should convert to for crewai

# '''
# ThirdPartyAgent(

# )
# '''
