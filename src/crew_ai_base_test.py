from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
import os

# Setup the required llm model to perform the tasks
os.environ["OPENAI_API_KEY"] = "NA"

llm = ChatOpenAI(model="llama3", base_url="http://localhost:11434/v1")

general_agent = Agent(
    role="Machine Learning Expert",
    goal="""Provide clear explanations to the questions asked in a very simple language covering even the complex concepts""",
    backstory="""You are an excellent Machiene Learning expert who explains complex concepts in a very understandable way""",
    allow_delegation=False,
    verbose=False,
    llm=llm,
)

task = Task(
    description="""what is K-Means Clustering""",
    agent=general_agent,
    expected_output="A short to meduim sized paragraph",
    rci_depth=1
)

crew = Crew(agents=[general_agent], tasks=[task], verbose=2)

result = crew.kickoff()

print(result)
