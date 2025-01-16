import random

from crewai import Agent, Crew, Task
from patronus import Client, EvaluationResult
from patronus_local_evaluator_tool import PatronusLocalEvaluatorTool

# Test the PatronusLocalEvaluatorTool where agent uses the local evaluator
client = Client()


# Example of an evaluator that returns a random pass/fail result
@client.register_local_evaluator("random_evaluator")
def random_evaluator(**kwargs):
    score = random.random()
    return EvaluationResult(
        score_raw=score,
        pass_=score >= 0.5,
        explanation="example explanation",  # Optional justification for LLM judges
    )


# 1. Uses PatronusEvalTool: agent can pick the best evaluator and criteria
# patronus_eval_tool = PatronusEvalTool()

# 2. Uses PatronusPredefinedCriteriaEvalTool: agent uses the defined evaluator and criteria
# patronus_eval_tool = PatronusPredefinedCriteriaEvalTool(
#     evaluators=[{"evaluator": "judge", "criteria": "contains-code"}]
# )

# 3. Uses PatronusLocalEvaluatorTool: agent uses user defined evaluator
patronus_eval_tool = PatronusLocalEvaluatorTool(
    patronus_client=client,
    evaluator="random_evaluator",
    evaluated_model_gold_answer="example label",
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
