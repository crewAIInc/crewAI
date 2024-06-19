from crewai import Task, Crew
from crewai.agents.third_party_agents.langchain_custom.agent import LangchainAgent

from langchain.agents import load_tools
from langchain_openai import OpenAI

llm = OpenAI(temperature=0)
langchain_tools = load_tools(["google-serper"], llm=llm)


agent1 = LangchainAgent(
    role="backstory agent",
    goal="who is {input}?",
    backstory="agent backstory",
    verbose=True,
    tools=langchain_tools,
)

task1 = Task(
    expected_output="a short biography of {input}",
    description="a short biography of {input}",
    agent=agent1,
)

agent2 = LangchainAgent(
    role="bio agent",
    goal="summarize the short bio for {input} and if needed do more research",
    backstory="agent backstory",
    verbose=True,
)

task2 = Task(
    description="a tldr summary of the short biography",
    expected_output="5 bullet point summary of the biography",
    agent=agent2,
    context=[task1],
)

my_crew = Crew(agents=[agent1, agent2], tasks=[task1, task2])
crew = my_crew.kickoff(inputs={"input": "andrew ng"})
print("crew", crew)
