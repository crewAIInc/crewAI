from crewai import Agent, Crew, Task
from patronus_eval_tool import PatronusEvalTool


patronus_eval_tool = PatronusEvalTool(
    evaluators=[{"evaluator": "judge", "criteria": "patronus:is-code"}], tags={}
)

# Create a new agent
coding_agent = Agent(
    role="Coding Agent",
    goal="Generate high quality code and verify that the code is correct by using Patronus AI's evaluation tool to check validity of your output code.",
    backstory="You are an experienced coder who can generate high quality python code. You can follow complex instructions accurately and effectively.",
    tools=[patronus_eval_tool],
    verbose=True,
)

# Define tasks
generate_code = Task(
    description="Create a simple program to generate the first N numbers in the Fibonacci sequence. Use the evaluator as `judge` from Patronus AI with the criteria `patronus:is-code` and feed your task input as input and your code as output to verify your code validity.",
    expected_output="Program that generates the first N numbers in the Fibonacci sequence.",
    agent=coding_agent,
)

crew = Crew(agents=[coding_agent], tasks=[generate_code])

crew.kickoff()
