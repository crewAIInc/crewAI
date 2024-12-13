import os

from crewai import Agent, Crew, Task
from patronus_eval_tool import PatronusEvalTool


patronus_eval_tool = PatronusEvalTool(
    evaluators=[{
        "evaluator": "judge",
        "criteria": "patronus:is-code"
    }],
    tags={}
)

# Create a new agent
coding_agent = Agent(
    role="Coding Agent",
    goal="Generate high quality code. Use the evaluation tool to score the agent outputs",
    backstory="Coding agent to generate high quality code. Use the evaluation tool to score the agent outputs",
    tools=[patronus_eval_tool],
    verbose=True,
)

# Define tasks
generate_code = Task(
    description="Create a simple program to generate the first N numbers in the Fibonacci sequence.",
    expected_output="Program that generates the first N numbers in the Fibonacci sequence.",
    agent=coding_agent,
)


crew = Crew(agents=[coding_agent], tasks=[generate_code])

crew.kickoff()