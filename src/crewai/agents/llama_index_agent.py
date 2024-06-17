from crewai import CustomAgent, Agent, Task, Crew, Process

import tiktoken
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import ReActAgent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI


# define sample Tool
def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b


def searchTool(search: str):
    """Uses DuckDuckGoSearchRun to search the web"""
    print("ADDING SEARCH IS", search)
    search = DuckDuckGoSearchRun()
    result = search.run(search)
    print("search result", result)
    return result


multiply_tool = FunctionTool.from_defaults(fn=multiply)
search_tool_llama = FunctionTool.from_defaults(fn=searchTool)

# initialize llm
llm = OpenAI(model="gpt-4o")


token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
)
llama_agent = ReActAgent.from_tools(
    [multiply_tool],
    llm=llm,
    verbose=True,
    callback_manager=CallbackManager([token_counter]),
)

# llama_agent.callback_manager = CallbackManager([token_counter])
llama_agent_2 = ReActAgent.from_tools([multiply_tool], llm=llm, verbose=True)
agent1 = CustomAgent(
    agent_executor=llama_agent.chat,
    token_counter=token_counter.completion_llm_token_count,
    role="math_solver",
    goal="solve the question",
    backstory="you are a mathmatician",
    verbose=True,
    memory=True,
    tools=[multiply_tool],
)

agent2 = Agent(
    role="funny_story",
    goal="create a funny story based on answer given",
    backstory="mathmatician turned commedian",
    verbose=True,
)


def callback_function(output):
    print(f"""
        Task completed!
        Task: {output.description}
        Output: {output.raw_output}
    """)


task1 = Task(
    agent=agent1,
    description="multiply the two numbers {a} and {b}.",
    expected_output="the multiplication of the two numbers given",
    callback=callback_function,
    tools=[multiply_tool],
)

task2 = Task(
    agent=agent2,
    description="make a joke about the math answer",
    expected_output="create a joke based on the answer",
    context=[task1],
)

# my_crew = Crew(agents=[agent1, agent2], tasks=[task1, task2], full_output=True)
my_crew = Crew(
    agents=[agent1, agent2],
    tasks=[task1, task2],
    full_output=True,
    process=Process.hierarchical,
    manager_llm=ChatOpenAI(temperature=0, model="gpt-4o"),
)
crew = my_crew.kickoff(inputs={"a": "32", "b": "25"})
crew = my_crew.kickoff(inputs={"a": "32", "b": "25"})
