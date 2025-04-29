import pytest
from unittest.mock import Mock, patch

from crewai import Agent, Task, TaskOutput


def test_combine_sub_task_results_no_sub_tasks():
    """Test that combining sub-task results raises an error when there are no sub-tasks."""
    agent = Agent(
        role="Researcher",
        goal="Research effectively",
        backstory="You're an expert researcher",
    )
    
    parent_task = Task(
        description="Research the impact of AI",
        expected_output="A report",
        agent=agent,
    )
    
    with pytest.raises(ValueError, match="Task has no sub-tasks to combine results from"):
        parent_task.combine_sub_task_results()


def test_combine_sub_task_results_no_agent():
    """Test that combining sub-task results raises an error when there is no agent."""
    parent_task = Task(
        description="Research the impact of AI",
        expected_output="A report",
    )
    
    sub_task = Task(
        description="Research AI impact on healthcare",
        expected_output="Healthcare report",
        parent_task=parent_task,
    )
    parent_task.sub_tasks.append(sub_task)
    
    with pytest.raises(ValueError, match="Task has no agent to combine sub-task results"):
        parent_task.combine_sub_task_results()


def test_execute_sync_sets_output_after_combining():
    """Test that execute_sync sets the output after combining sub-task results."""
    agent = Agent(
        role="Researcher",
        goal="Research effectively",
        backstory="You're an expert researcher",
    )
    
    parent_task = Task(
        description="Research the impact of AI",
        expected_output="A report",
        agent=agent,
    )
    
    sub_tasks = parent_task.decompose([
        "Research AI impact on healthcare",
        "Research AI impact on finance",
    ])
    
    with patch.object(Agent, 'execute_task', return_value="Combined result") as mock_execute_task:
        result = parent_task.execute_sync()
        
        assert parent_task.output is not None
        assert parent_task.output.raw == "Combined result"
        assert result.raw == "Combined result"
        
        assert mock_execute_task.call_count >= 3


def test_deep_cloning_prevents_shared_state():
    """Test that deep cloning prevents shared mutable state between tasks."""
    agent = Agent(
        role="Researcher",
        goal="Research effectively",
        backstory="You're an expert researcher",
    )
    
    parent_task = Task(
        description="Research the impact of AI",
        expected_output="A report",
        agent=agent,
    )
    
    copied_task = parent_task.copy()
    
    copied_task.description = "Modified description"
    
    assert parent_task.description == "Research the impact of AI"
    assert copied_task.description == "Modified description"
    
    parent_task.decompose(["Sub-task 1", "Sub-task 2"])
    
    assert len(parent_task.sub_tasks) == 2
    assert len(copied_task.sub_tasks) == 0


def test_execute_sub_tasks_async_empty_sub_tasks():
    """Test that execute_sub_tasks_async returns an empty list when there are no sub-tasks."""
    parent_task = Task(
        description="Research the impact of AI",
        expected_output="A report",
    )
    
    futures = parent_task.execute_sub_tasks_async()
    
    assert isinstance(futures, list)
    assert len(futures) == 0
