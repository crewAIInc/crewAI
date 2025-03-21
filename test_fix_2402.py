# test_fix_2402.py
from crewai import Agent, Crew, Task

# Case 1: Only system_template provided
agent1 = Agent(
    role="Test Role",
    goal="Test Goal",
    backstory="Test Backstory",
    system_template="You are a test agent...",
    # prompt_template is intentionally missing
)

# Case 2: system_template and prompt_template provided, but response_template missing
agent2 = Agent(
    role="Test Role",
    goal="Test Goal",
    backstory="Test Backstory",
    system_template="You are a test agent...",
    prompt_template="This is a test prompt...",
    # response_template is intentionally missing
)

# Create tasks and crews
task1 = Task(description="Test task 1", agent=agent1, expected_output="Test output 1")
task2 = Task(description="Test task 2", agent=agent2, expected_output="Test output 2")

crew1 = Crew(agents=[agent1], tasks=[task1])
crew2 = Crew(agents=[agent2], tasks=[task2])

print("Testing agent with only system_template...")
try:
    agent1.execute_task(task1)
    print("Success! No error was raised.")
except Exception as e:
    print(f"Error: {e}")

print("\nTesting agent with missing response_template...")
try:
    agent2.execute_task(task2)
    print("Success! No error was raised.")
except Exception as e:
    print(f"Error: {e}")
