from crewai import Agent, Crew, Task
from patronus_eval_tool import (
    PatronusEvalTool,
    PatronusPredifinedCriteriaEvalTool,
    PatronusLocalEvaluatorTool,
)
from patronus import Client, EvaluationResult

# Test the PatronusEvalTool where agent can pick the best evaluator and criteria
patronus_eval_tool = PatronusEvalTool()

# Test the PatronusPredifinedCriteriaEvalTool where agent uses the defined evaluator and criteria
patronus_eval_tool = PatronusPredifinedCriteriaEvalTool(
    evaluators=[{"evaluator": "judge", "criteria": "contains-code"}]
)

# Test the PatronusLocalEvaluatorTool where agent uses the local evaluator
client = Client()


@client.register_local_evaluator("local_evaluator_name")
def my_evaluator(**kwargs):
    return EvaluationResult(pass_="PASS", score=0.5, explanation="Explanation test")


patronus_eval_tool = PatronusLocalEvaluatorTool(
    evaluator="local_evaluator_name", evaluated_model_gold_answer="test"
)


# Create a new agent
coding_agent = Agent(
    role="Coding Agent",
    goal="Generate high quality code and verify that the output is code by using Patronus AI's evaluation tool.",
    backstory="You are an experienced coder who can generate high quality python code. You can follow complex instructions accurately and effectively.",
    tools=[patronus_eval_tool],
    verbose=True,
)

# Define tasks
generate_code = Task(
    description="Create a simple program to generate the first N numbers in the Fibonacci sequence. Select the most appropriate evaluator and criteria for evaluating your output.",
    expected_output="Program that generates the first N numbers in the Fibonacci sequence.",
    agent=coding_agent,
)

crew = Crew(agents=[coding_agent], tasks=[generate_code])

crew.kickoff()
