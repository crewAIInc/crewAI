from llama_index.core.tools import FunctionTool
# from llama_index.llms.openai import OpenAI
# from llama_index.core.agent import ReActAgent

from llama_index.llms.openai import OpenAI
from llama_index.core.agent import ReActAgent
from crewai import CustomAgentWrapper, Agent, Task, Crew
from crewai.utilities import Printer

# from crewai_tools import SerperDevTool
from langchain_community.tools import DuckDuckGoSearchRun


printer = Printer()


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
# print('search_tool_llama',search_tool_llama.call(search='things to do in los angeles'))

# searchTool = FunctionTool.from_defaults(fn=multiply)
# duckduckgo_instant_search(query: str) -> List[Dict]

# initialize llm
llm = OpenAI(model="gpt-3.5-turbo-0613")

# initialize ReAct agent
llama_agent = ReActAgent.from_tools([multiply_tool], llm=llm, verbose=True)

# print('llama_index_agent_AgentRunner',llama_index_agent)
# response = llama_index_step.chat('what is 2 * 5')
# print('response', response)
agent1 = CustomAgentWrapper(
    custom_agent=llama_agent,
    agent_executor=llama_agent.chat,
    role="Math solver",
    goal="solve the question",
    backstory="you are a mathmatician",
    verbose=True,
    memory=True,
    tools=[multiply_tool],
)
# agent1 = CustomAgentWrapper(
#     custom_agent=llama_agent,
#     agent_executor=llama_agent.chat,
#     role="Event Finder",
#     goal="find events in {input}",
#     backstory="You are a expert date planner",
#     verbose=True
#     )
# print("---AGENT 1---")
# printer.print(vars(agent1), color='bold_green')


agent2 = Agent(
    # custom_agent=llama_index_agent,
    # agent_executor=AgentRunner.,
    role="funny story",
    goal="create a funny story based on answer given",
    backstory="mathmatician turned commedian",
    verbose=True,
)
# print("---agent2---")
# printer.print(vars(agent2), color='bold_green')
task1 = Task(
    agent=agent1,
    description="multiply the two numbers {a} and {b}.",
    expected_output="the multiplication of the two numbers given",
)
# task1 = Task(
#     agent=agent1,
#     description="multiply the two numbers",
#     expected_output="give a bullet point list of things to do in {input}.",
# )
task2 = Task(
    agent=agent2,
    description="make a joke about the math answer",
    expected_output="create a joke based on the answer",
    context=[task1],
)

# print("---taask 1---")
# printer.print(vars(task1), color='bold_purple')
# print("---taask 2---")
# printer.print(vars(task2), color='bold_purple')

# my_crew = Crew(
#     agents=[agent1],
#     tasks=[task1],

# )
my_crew = Crew(
    agents=[agent1, agent2],
    tasks=[task1, task2],
)
# print("my_crew",my_crew)
crew = my_crew.kickoff(inputs={"a": "2", "b": "3"})
# crew = my_crew.kickoff(inputs={'input': 'things to do in los angeles'})
print("crew", crew)
