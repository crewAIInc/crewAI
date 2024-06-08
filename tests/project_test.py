from crewai.agent import Agent
from crewai.project import agent, task
from crewai.task import Task


class SimpleCrew:
    @agent
    def simple_agent(self):
        return Agent(
            role="Simple Agent", goal="Simple Goal", backstory="Simple Backstory"
        )

    @task
    def simple_task(self):
        return Task(description="Simple Description", expected_output="Simple Output")


def test_agent_memoization():
    crew = SimpleCrew()
    first_call_result = crew.simple_agent()
    second_call_result = crew.simple_agent()

    assert (
        first_call_result is second_call_result
    ), "Agent memoization is not working as expected"


def test_task_memoization():
    crew = SimpleCrew()
    first_call_result = crew.simple_task()
    second_call_result = crew.simple_task()

    assert (
        first_call_result is second_call_result
    ), "Task memoization is not working as expected"
