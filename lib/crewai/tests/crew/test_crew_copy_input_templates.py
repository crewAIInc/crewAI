"""Copies must keep the input templates of tasks and agents."""

from crewai.agent import Agent
from crewai.crew import Crew
from crewai.task import Task


def _crew() -> tuple[Crew, Task, Agent]:
    agent = Agent(
        role="Researcher of {topic}",
        goal="Research {topic}",
        backstory="An expert on {topic}.",
        llm="gpt-4o-mini",
    )
    task = Task(
        description="Research {topic}.",
        expected_output="Notes about {topic}.",
        agent=agent,
    )
    return Crew(agents=[agent], tasks=[task]), task, agent


def test_copied_task_reinterpolates_with_the_new_inputs() -> None:
    crew, task, _ = _crew()
    crew._interpolate_inputs({"topic": "alpha"})
    assert task.description == "Research alpha."

    copied = crew.copy()
    copied_task = copied.tasks[0]
    assert copied_task._original_description == "Research {topic}."
    assert copied_task._original_expected_output == "Notes about {topic}."

    copied._interpolate_inputs({"topic": "beta"})
    assert copied_task.description == "Research beta."
    assert copied_task.expected_output == "Notes about beta."
    # The source keeps the run it already interpolated.
    assert task.description == "Research alpha."


def test_copied_agent_reinterpolates_with_the_new_inputs() -> None:
    crew, _, agent = _crew()
    crew._interpolate_inputs({"topic": "alpha"})
    assert agent.role == "Researcher of alpha"

    copied = crew.copy()
    copied_agent = copied.agents[0]

    copied._interpolate_inputs({"topic": "beta"})
    assert copied_agent.role == "Researcher of beta"
    assert copied_agent.goal == "Research beta"
    assert copied_agent.backstory == "An expert on beta."
    assert agent.role == "Researcher of alpha"


def test_copy_before_any_interpolation_leaves_templates_unset() -> None:
    crew, task, agent = _crew()

    copied = crew.copy()

    assert copied.tasks[0].description == task.description
    assert copied.agents[0].role == agent.role
    assert copied.tasks[0]._original_description is None
