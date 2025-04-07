"""Test for multiple conditional tasks."""

from unittest.mock import MagicMock, patch

import pytest

from crewai.agent import Agent
from crewai.crew import Crew
from crewai.task import Task
from crewai.tasks.conditional_task import ConditionalTask
from crewai.tasks.output_format import OutputFormat
from crewai.tasks.task_output import TaskOutput


def test_multiple_conditional_tasks():
    """Test that multiple conditional tasks are evaluated correctly."""
    # Create agents for the tasks
    agent1 = Agent(
        role="Research Analyst",
        goal="Find information",
        backstory="You're a researcher",
        verbose=True,
    )
    
    agent2 = Agent(
        role="Data Analyst",
        goal="Process information",
        backstory="You process data",
        verbose=True,
    )
    
    agent3 = Agent(
        role="Report Writer",
        goal="Write reports",
        backstory="You write reports",
        verbose=True,
    )
    
    # Create tasks
    task1 = Task(
        description="Task 1",
        expected_output="Output 1",
        agent=agent1,
    )
    
    # First conditional task should check task1's output
    condition1_mock = MagicMock()
    task2 = ConditionalTask(
        description="Conditional Task 2",
        expected_output="Output 2",
        agent=agent2,
        condition=condition1_mock,
    )
    
    # Second conditional task should check task1's output, not task2's
    condition2_mock = MagicMock()
    task3 = ConditionalTask(
        description="Conditional Task 3",
        expected_output="Output 3",
        agent=agent3,
        condition=condition2_mock,
    )
    
    # Create crew with the tasks
    crew = Crew(
        agents=[agent1, agent2, agent3],
        tasks=[task1, task2, task3],
        verbose=True,
    )
    
    # Test the first conditional task
    task1_output = TaskOutput(
        description="Task 1",
        raw="Task 1 output",
        agent=agent1.role,
        output_format=OutputFormat.RAW,
    )
    
    condition1_mock.return_value = True  # Task should execute
    result1 = crew._handle_conditional_task(
        task=task2,
        task_outputs=[task1_output],
        futures=[],
        task_index=1,
        was_replayed=False,
    )
    
    # Verify the condition was called with task1's output
    condition1_mock.assert_called_once()
    args1 = condition1_mock.call_args[0][0]
    assert args1.raw == "Task 1 output"
    assert result1 is None  # Task should execute, so no skipped output
    
    # Test the second conditional task
    task2_output = TaskOutput(
        description="Conditional Task 2",
        raw="Task 2 output",
        agent=agent2.role,
        output_format=OutputFormat.RAW,
    )
    
    condition2_mock.return_value = True  # Task should execute
    result2 = crew._handle_conditional_task(
        task=task3,
        task_outputs=[task1_output, task2_output],
        futures=[],
        task_index=2,
        was_replayed=False,
    )
    
    # Verify the condition was called with task1's output, not task2's
    condition2_mock.assert_called_once()
    args2 = condition2_mock.call_args[0][0]
    assert args2.raw == "Task 1 output"  # Should be task1's output
    assert args2.raw != "Task 2 output"  # Should not be task2's output
    assert result2 is None  # Task should execute, so no skipped output
