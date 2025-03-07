from unittest.mock import MagicMock, patch

import pytest

from crewai import Agent, Task
from crewai.flow import Flow, listen, start
from crewai.project.annotations import task
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


class TestFlowWithConditionalTasks(Flow):
    """Test flow with conditional tasks."""

    @start()
    @task
    def initial_task(self):
        """Initial task that always executes."""
        return Task(
            description="Initial task",
            expected_output="Initial output",
            agent=researcher,
        )

    @listen(initial_task)
    @task
    def conditional_task_false(self):
        """Conditional task that should be skipped."""
        return ConditionalTask(
            description="Conditional task that should be skipped",
            expected_output="This should not be executed",
            agent=writer,
            condition=False,
        )

    @listen(initial_task)
    @task
    def conditional_task_true(self):
        """Conditional task that should be executed."""
        return ConditionalTask(
            description="Conditional task that should be executed",
            expected_output="This should be executed",
            agent=writer,
            condition=True,
        )

    @listen(conditional_task_true)
    @task
    def final_task(self):
        """Final task that executes after the conditional task."""
        return Task(
            description="Final task",
            expected_output="Final output",
            agent=researcher,
        )


def test_flow_with_conditional_tasks():
    """Test that conditional tasks work correctly in a Flow."""
    flow = TestFlowWithConditionalTasks()

    with patch.object(Task, "execute_sync") as mock_execute_sync:
        mock_execute_sync.return_value = TaskOutput(
            description="Task output",
            raw="Task output",
            agent="Agent",
        )

        flow.kickoff()

        # The initial task, conditional_task_true, and final_task should be executed
        # conditional_task_false should be skipped
        assert mock_execute_sync.call_count == 3


class TestFlowWithSequentialConditionalTasks(Flow):
    """Test flow with sequential conditional tasks."""

    @start()
    @task
    def initial_task(self):
        """Initial task that always executes."""
        return Task(
            description="Initial task",
            expected_output="Initial output",
            agent=researcher,
        )

    @listen(initial_task)
    @task
    def conditional_task_1(self):
        """First conditional task that should be executed."""
        return ConditionalTask(
            description="First conditional task",
            expected_output="First conditional output",
            agent=writer,
            condition=True,
        )

    @listen(conditional_task_1)
    @task
    def conditional_task_2(self):
        """Second conditional task that should be skipped."""
        return ConditionalTask(
            description="Second conditional task",
            expected_output="Second conditional output",
            agent=researcher,
            condition=False,
        )

    @listen(conditional_task_2)
    @task
    def conditional_task_3(self):
        """Third conditional task that should be executed."""
        return ConditionalTask(
            description="Third conditional task",
            expected_output="Third conditional output",
            agent=writer,
            condition=True,
        )


def test_flow_with_sequential_conditional_tasks():
    """Test that sequential conditional tasks work correctly in a Flow."""
    flow = TestFlowWithSequentialConditionalTasks()

    with patch.object(Task, "execute_sync") as mock_execute_sync:
        mock_execute_sync.return_value = TaskOutput(
            description="Task output",
            raw="Task output",
            agent="Agent",
        )

        flow.kickoff()

        # The initial_task and conditional_task_1 should be executed
        # conditional_task_2 should be skipped, and since it's skipped,
        # conditional_task_3 should not be triggered
        assert mock_execute_sync.call_count == 2
