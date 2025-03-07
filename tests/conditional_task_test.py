from unittest.mock import MagicMock, patch

import pytest

from crewai import Agent, Crew, Task
from crewai.tasks.conditional_task import ConditionalTask
from crewai.tasks.task_output import TaskOutput

# Create mock agents for testing
researcher = Agent(
    role="Researcher",
    goal="Research information",
    backstory="You are a researcher with expertise in finding information.",
)

writer = Agent(
    role="Writer",
    goal="Write content",
    backstory="You are a writer with expertise in creating engaging content.",
)


def test_conditional_task_with_boolean_false():
    """Test that a conditional task with a boolean False condition is skipped."""
    task1 = Task(
        description="Initial task",
        expected_output="Initial output",
        agent=researcher,
    )

    # Use a boolean False directly as the condition
    task2 = ConditionalTask(
        description="Conditional task that should be skipped",
        expected_output="This should not be executed",
        agent=writer,
        condition=False,
    )

    crew = Crew(
        agents=[researcher, writer],
        tasks=[task1, task2],
    )

    with patch.object(Task, "execute_sync") as mock_execute_sync:
        mock_execute_sync.return_value = TaskOutput(
            description="Task 1 description",
            raw="Task 1 output",
            agent="Researcher",
        )

        result = crew.kickoff()

        # Only the first task should be executed
        assert mock_execute_sync.call_count == 1

        # The conditional task should be skipped
        assert task2.output is not None
        assert task2.output.raw == ""

        # The final output should be from the first task
        assert result.raw.startswith("Task 1 output")


def test_conditional_task_with_boolean_true():
    """Test that a conditional task with a boolean True condition is executed."""
    task1 = Task(
        description="Initial task",
        expected_output="Initial output",
        agent=researcher,
    )

    # Use a boolean True directly as the condition
    task2 = ConditionalTask(
        description="Conditional task that should be executed",
        expected_output="This should be executed",
        agent=writer,
        condition=True,
    )

    crew = Crew(
        agents=[researcher, writer],
        tasks=[task1, task2],
    )

    with patch.object(Task, "execute_sync") as mock_execute_sync:
        mock_execute_sync.return_value = TaskOutput(
            description="Task output",
            raw="Task output",
            agent="Agent",
        )

        crew.kickoff()

        # Both tasks should be executed
        assert mock_execute_sync.call_count == 2


def test_multiple_sequential_conditional_tasks():
    """Test that multiple conditional tasks in sequence work correctly."""
    task1 = Task(
        description="Initial task",
        expected_output="Initial output",
        agent=researcher,
    )

    # First conditional task (will be executed)
    task2 = ConditionalTask(
        description="First conditional task",
        expected_output="First conditional output",
        agent=writer,
        condition=True,
    )

    # Second conditional task (will be skipped)
    task3 = ConditionalTask(
        description="Second conditional task",
        expected_output="Second conditional output",
        agent=researcher,
        condition=False,
    )

    # Third conditional task (will be executed)
    task4 = ConditionalTask(
        description="Third conditional task",
        expected_output="Third conditional output",
        agent=writer,
        condition=True,
    )

    crew = Crew(
        agents=[researcher, writer],
        tasks=[task1, task2, task3, task4],
    )

    with patch.object(Task, "execute_sync") as mock_execute_sync:
        mock_execute_sync.return_value = TaskOutput(
            description="Task output",
            raw="Task output",
            agent="Agent",
        )

        result = crew.kickoff()

        # Tasks 1, 2, and 4 should be executed (task 3 is skipped)
        assert mock_execute_sync.call_count == 3

        # Task 3 should be skipped
        assert task3.output is not None
        assert task3.output.raw == ""


def test_last_task_conditional():
    """Test that a conditional task at the end of the task list works correctly."""
    task1 = Task(
        description="Initial task",
        expected_output="Initial output",
        agent=researcher,
    )

    # Last task is conditional and will be skipped
    task2 = ConditionalTask(
        description="Last conditional task",
        expected_output="Last conditional output",
        agent=writer,
        condition=False,
    )

    crew = Crew(
        agents=[researcher, writer],
        tasks=[task1, task2],
    )

    with patch.object(Task, "execute_sync") as mock_execute_sync:
        mock_execute_sync.return_value = TaskOutput(
            description="Task 1 output",
            raw="Task 1 output",
            agent="Researcher",
        )

        result = crew.kickoff()

        # Only the first task should be executed
        assert mock_execute_sync.call_count == 1

        # The conditional task should be skipped
        assert task2.output is not None
        assert task2.output.raw == ""

        # The final output should be from the first task
        assert result.raw.startswith("Task 1 output")
