# test.py
from crewai import Task, Crew, Process
from crewai.agents.agent_adapters.foundry_agents.foundry_adapter import FoundryAgentAdapter
from crewai_tools import SerperDevTool
import os

# Create a single agent
agent = FoundryAgentAdapter(
    role="Quick Assistant",
    goal="Answer simple questions clearly",
    backstory="You're a helpful assistant",
    verbose=True,
    tools=[SerperDevTool()],
)

# Define a task
task = Task(
    description="Answer the question: 'What is the capital of France?'",
    expected_output="The capital of France is Paris.",
    agent=agent,
)

# Create a crew to run it
crew = Crew(
    agents=[agent],
    tasks=[task],
    process=Process.sequential,
    verbose=True,
)

# Run the crew
if __name__ == "__main__":
    result = crew.kickoff()
    print("Result:", result)
