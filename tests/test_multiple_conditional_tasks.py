"""Test for multiple conditional tasks."""

from unittest.mock import MagicMock, patch

import pytest

from crewai.agent import Agent
from crewai.crew import Crew
from crewai.task import Task
from crewai.tasks.conditional_task import ConditionalTask
from crewai.tasks.output_format import OutputFormat
from crewai.tasks.task_output import TaskOutput


class TestMultipleConditionalTasks:
    """Test class for multiple conditional tasks scenarios."""

    @pytest.fixture
    def setup_agents(self):
        """Set up agents for the tests."""
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
        
        return agent1, agent2, agent3
    
    @pytest.fixture
    def setup_tasks(self, setup_agents):
        """Set up tasks for the tests."""
        agent1, agent2, agent3 = setup_agents
        
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
        
        return task1, task2, task3, condition1_mock, condition2_mock
    
    @pytest.fixture
    def setup_crew(self, setup_agents, setup_tasks):
        """Set up crew for the tests."""
        agent1, agent2, agent3 = setup_agents
        task1, task2, task3, _, _ = setup_tasks
        
        crew = Crew(
            agents=[agent1, agent2, agent3],
            tasks=[task1, task2, task3],
            verbose=True,
        )
        
        return crew
    
    @pytest.fixture
    def setup_task_outputs(self, setup_agents):
        """Set up task outputs for the tests."""
        agent1, agent2, _ = setup_agents
        
        task1_output = TaskOutput(
            description="Task 1",
            raw="Task 1 output",
            agent=agent1.role,
            output_format=OutputFormat.RAW,
        )
        
        task2_output = TaskOutput(
            description="Conditional Task 2",
            raw="Task 2 output",
            agent=agent2.role,
            output_format=OutputFormat.RAW,
        )
        
        return task1_output, task2_output
    
    def test_first_conditional_task_execution(self, setup_crew, setup_tasks, setup_task_outputs):
        """Test that the first conditional task is evaluated correctly."""
        crew = setup_crew
        _, task2, _, condition1_mock, _ = setup_tasks
        task1_output, _ = setup_task_outputs
        
        condition1_mock.return_value = True  # Task should execute
        result = crew._handle_conditional_task(
            task=task2,
            task_outputs=[task1_output],
            futures=[],
            task_index=1,
            was_replayed=False,
        )
        
        # Verify the condition was called with task1's output
        condition1_mock.assert_called_once()
        args = condition1_mock.call_args[0][0]
        assert args.raw == "Task 1 output"
        assert result is None  # Task should execute, so no skipped output
    
    def test_second_conditional_task_execution(self, setup_crew, setup_tasks, setup_task_outputs):
        """Test that the second conditional task is evaluated correctly."""
        crew = setup_crew
        _, _, task3, _, condition2_mock = setup_tasks
        task1_output, task2_output = setup_task_outputs
        
        condition2_mock.return_value = True  # Task should execute
        result = crew._handle_conditional_task(
            task=task3,
            task_outputs=[task1_output, task2_output],
            futures=[],
            task_index=2,
            was_replayed=False,
        )
        
        # Verify the condition was called with task1's output, not task2's
        condition2_mock.assert_called_once()
        args = condition2_mock.call_args[0][0]
        assert args.raw == "Task 1 output"  # Should be task1's output
        assert args.raw != "Task 2 output"  # Should not be task2's output
        assert result is None  # Task should execute, so no skipped output
    
    def test_conditional_task_skipping(self, setup_crew, setup_tasks, setup_task_outputs):
        """Test that conditional tasks are skipped when the condition returns False."""
        crew = setup_crew
        _, task2, _, condition1_mock, _ = setup_tasks
        task1_output, _ = setup_task_outputs
        
        condition1_mock.return_value = False  # Task should be skipped
        result = crew._handle_conditional_task(
            task=task2,
            task_outputs=[task1_output],
            futures=[],
            task_index=1,
            was_replayed=False,
        )
        
        # Verify the condition was called with task1's output
        condition1_mock.assert_called_once()
        args = condition1_mock.call_args[0][0]
        assert args.raw == "Task 1 output"
        assert result is not None  # Task should be skipped, so there should be a skipped output
        assert result.description == task2.description
    
    def test_conditional_task_with_explicit_context(self, setup_crew, setup_agents, setup_task_outputs):
        """Test conditional task with explicit context tasks."""
        crew = setup_crew
        agent1, agent2, _ = setup_agents
        task1_output, _ = setup_task_outputs
        
        with patch.object(crew, '_find_task_index', return_value=0):
            context_task = Task(
                description="Task 1",
                expected_output="Output 1",
                agent=agent1,
            )
            
            condition_mock = MagicMock(return_value=True)
            task_with_context = ConditionalTask(
                description="Task with Context",
                expected_output="Output with Context",
                agent=agent2,
                condition=condition_mock,
                context=[context_task],
            )
            
            crew.tasks.append(task_with_context)
            
            result = crew._handle_conditional_task(
                task=task_with_context,
                task_outputs=[task1_output],
                futures=[],
                task_index=3,  # This would be the 4th task
                was_replayed=False,
            )
        
        # Verify the condition was called with task1's output
        condition_mock.assert_called_once()
        args = condition_mock.call_args[0][0]
        assert args.raw == "Task 1 output"
        assert result is None  # Task should execute, so no skipped output
    
    def test_conditional_task_with_empty_task_outputs(self, setup_crew, setup_tasks):
        """Test conditional task with empty task outputs."""
        crew = setup_crew
        _, task2, _, condition1_mock, _ = setup_tasks
        
        result = crew._handle_conditional_task(
            task=task2,
            task_outputs=[],
            futures=[],
            task_index=1,
            was_replayed=False,
        )
        
        condition1_mock.assert_not_called()
        assert result is None  # Task should execute, so no skipped output


def test_multiple_conditional_tasks():
    """Test that multiple conditional tasks are evaluated correctly.
    
    This is a legacy test that's kept for backward compatibility.
    The actual tests are now in the TestMultipleConditionalTasks class.
    """
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
    
    crew = Crew(
        agents=[agent1, agent2, agent3],
        tasks=[task1, task2, task3],
        verbose=True,
    )
    
    with patch.object(crew, '_find_task_index', return_value=0):
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
        
        condition1_mock.reset_mock()
        
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
