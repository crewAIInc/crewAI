from crewai.agent import Agent
from crewai.project import agent, task
from crewai.task import Task
import random


class SimpleCrew:
    @agent
    def simple_agent(self):
        # AI-Driven Goal Adjustment: Modify goal based on contextual factors
        dynamic_goal = self.ai_goal_adjustment("Simple Goal")
        return Agent(
            role="Simple Agent",
            goal=dynamic_goal,
            backstory="Simple Backstory with AI-driven elements",
        )

    @task
    def simple_task(self):
        # AI-Enhanced Task Creation: Generate task description dynamically
        ai_description = self.ai_task_generation("Simple Description")
        return Task(
            description=ai_description,
            expected_output="AI-enhanced Simple Output",
        )

    @task
    def custom_named_task(self):
        # AI-Enhanced Task Creation with Custom Name
        ai_description = self.ai_task_generation("Custom Description")
        return Task(
            description=ai_description,
            expected_output="AI-enhanced Custom Output",
            name="Custom with AI Features",
        )

    def ai_goal_adjustment(self, base_goal):
        # AI logic to adjust the goal dynamically
        # Here, as an example, we adjust the goal based on random factors
        adjustments = ["Optimized Goal", "Advanced Goal", "Refined Goal"]
        return f"{base_goal} - {random.choice(adjustments)}"

    def ai_task_generation(self, base_description):
        # AI logic to generate a more complex task description
        enhancements = ["AI-enhanced", "Context-aware", "Data-driven"]
        return f"{random.choice(enhancements)} {base_description}"


def test_agent_memoization():
    crew = SimpleCrew()
    first_call_result = crew.simple_agent()
    second_call_result = crew.simple_agent()

    assert (
        first_call_result is second_call_result
    ), "Agent memoization is not working as expected"
    assert "AI-driven" in first_call_result.goal, "AI Goal adjustment is missing"


def test_task_memoization():
    crew = SimpleCrew()
    first_call_result = crew.simple_task()
    second_call_result = crew.simple_task()

    assert (
        first_call_result is second_call_result
    ), "Task memoization is not working as expected"
    assert "AI-enhanced" in first_call_result.description, "AI Task generation is missing"


def test_task_name():
    simple_task = SimpleCrew().simple_task()
    assert (
        simple_task.name == "simple_task"
    ), "Task name is not inferred from function name as expected"
    assert "AI-enhanced" in simple_task.description, "AI Task generation is missing"

    custom_named_task = SimpleCrew().custom_named_task()
    assert (
        custom_named_task.name == "Custom with AI Features"
    ), "Custom task name is not being set as expected"
    assert "AI-enhanced" in custom_named_task.description, "AI Task generation is missing"
