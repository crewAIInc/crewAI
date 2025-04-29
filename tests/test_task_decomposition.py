import pytest
from unittest.mock import Mock, patch

from crewai import Agent, Task


def test_task_decomposition_structure():
    """Test that task decomposition creates the proper parent-child relationship."""
    agent = Agent(
        role="Researcher",
        goal="Research effectively",
        backstory="You're an expert researcher",
    )
    
    parent_task = Task(
        description="Research the impact of AI on various industries",
        expected_output="A comprehensive report",
        agent=agent,
    )
    
    sub_task_descriptions = [
        "Research AI impact on healthcare",
        "Research AI impact on finance",
        "Research AI impact on education",
    ]
    
    sub_tasks = parent_task.decompose(
        descriptions=sub_task_descriptions,
        expected_outputs=["Healthcare report", "Finance report", "Education report"],
        names=["Healthcare", "Finance", "Education"],
    )
    
    assert len(sub_tasks) == 3
    assert len(parent_task.sub_tasks) == 3
    
    for sub_task in sub_tasks:
        assert sub_task.parent_task == parent_task
        assert parent_task in sub_task.context


def test_task_execution_with_sub_tasks():
    """Test that executing a task with sub-tasks executes the sub-tasks first."""
    agent = Agent(
        role="Researcher",
        goal="Research effectively",
        backstory="You're an expert researcher",
    )
    
    parent_task = Task(
        description="Research the impact of AI on various industries",
        expected_output="A comprehensive report",
        agent=agent,
    )
    
    sub_task_descriptions = [
        "Research AI impact on healthcare",
        "Research AI impact on finance",
        "Research AI impact on education",
    ]
    
    parent_task.decompose(
        descriptions=sub_task_descriptions,
        expected_outputs=["Healthcare report", "Finance report", "Education report"],
    )
    
    with patch.object(Agent, 'execute_task', return_value="Mock result") as mock_execute_task:
        result = parent_task.execute_sync()
        
        assert mock_execute_task.call_count >= 3
        
        for sub_task in parent_task.sub_tasks:
            assert sub_task.output is not None
        
        assert result is not None
        assert result.raw is not None


def test_combine_sub_task_results():
    """Test that combining sub-task results works correctly."""
    agent = Agent(
        role="Researcher",
        goal="Research effectively",
        backstory="You're an expert researcher",
    )
    
    parent_task = Task(
        description="Research the impact of AI on various industries",
        expected_output="A comprehensive report",
        agent=agent,
    )
    
    sub_tasks = parent_task.decompose([
        "Research AI impact on healthcare",
        "Research AI impact on finance",
    ])
    
    for sub_task in sub_tasks:
        sub_task.output = Mock()
        sub_task.output.raw = f"Result for {sub_task.description}"
    
    with patch.object(Agent, 'execute_task', return_value="Combined result") as mock_execute_task:
        result = parent_task.combine_sub_task_results()
        
        assert mock_execute_task.called
        assert result == "Combined result"


def test_task_decomposition_validation():
    """Test that task decomposition validates inputs correctly."""
    parent_task = Task(
        description="Research the impact of AI",
        expected_output="A report",
    )
    
    with pytest.raises(ValueError, match="At least one sub-task description is required"):
        parent_task.decompose([])
    
    with pytest.raises(ValueError, match="expected_outputs must have the same length"):
        parent_task.decompose(
            ["Task 1", "Task 2"], 
            expected_outputs=["Output 1"]
        )
    
    with pytest.raises(ValueError, match="names must have the same length"):
        parent_task.decompose(
            ["Task 1", "Task 2"], 
            names=["Name 1"]
        )


def test_execute_sub_tasks_async():
    """Test that executing sub-tasks asynchronously works correctly."""
    agent = Agent(
        role="Researcher",
        goal="Research effectively",
        backstory="You're an expert researcher",
    )
    
    parent_task = Task(
        description="Research the impact of AI on various industries",
        expected_output="A comprehensive report",
        agent=agent,
    )
    
    sub_tasks = parent_task.decompose([
        "Research AI impact on healthcare",
        "Research AI impact on finance",
    ])
    
    with patch.object(Task, 'execute_async') as mock_execute_async:
        mock_future = Mock()
        mock_execute_async.return_value = mock_future
        
        futures = parent_task.execute_sub_tasks_async()
        
        assert mock_execute_async.call_count == 2
        assert len(futures) == 2
