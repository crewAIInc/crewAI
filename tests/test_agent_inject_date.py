from datetime import datetime
from unittest.mock import patch

from crewai.agent import Agent
from crewai.task import Task


def test_agent_inject_date():
    """Test that the inject_date flag injects the current date into the task.
    
    Tests that when inject_date=True, the current date is added to the task description.
    """
    agent = Agent(
        role="test_agent",
        goal="test_goal",
        backstory="test_backstory",
        inject_date=True,
    )
    
    task = Task(
        description="Test task",
        expected_output="Test output",
        agent=agent,
    )
    
    with patch.object(Agent, 'execute_task', return_value="Task executed") as mock_execute:
        agent.execute_task(task)
        
        called_task = mock_execute.call_args[0][0]
        
        current_date = datetime.now().strftime("%Y-%m-%d")
        assert f"Current Date: {current_date}" in called_task.description


def test_agent_without_inject_date():
    """Test that without inject_date flag, no date is injected.
    
    Tests that when inject_date=False (default), no date is added to the task description.
    """
    agent = Agent(
        role="test_agent",
        goal="test_goal",
        backstory="test_backstory",
    )
    
    task = Task(
        description="Test task",
        expected_output="Test output",
        agent=agent,
    )
    
    original_description = task.description
    
    with patch.object(Agent, 'execute_task', return_value="Task executed") as mock_execute:
        agent.execute_task(task)
        
        called_task = mock_execute.call_args[0][0]
        
        assert "Current Date:" not in called_task.description
        assert called_task.description == original_description


def test_agent_inject_date_custom_format():
    """Test that the inject_date flag with custom date_format works correctly.
    
    Tests that when inject_date=True with a custom date_format, the date is formatted correctly.
    """
    agent = Agent(
        role="test_agent",
        goal="test_goal",
        backstory="test_backstory",
        inject_date=True,
        date_format="%d/%m/%Y",
    )
    
    task = Task(
        description="Test task",
        expected_output="Test output",
        agent=agent,
    )
    
    with patch.object(Agent, 'execute_task', return_value="Task executed") as mock_execute:
        agent.execute_task(task)
        
        called_task = mock_execute.call_args[0][0]
        
        current_date = datetime.now().strftime("%d/%m/%Y")
        assert f"Current Date: {current_date}" in called_task.description


def test_agent_inject_date_invalid_format():
    """Test error handling with invalid date format.
    
    Tests that when an invalid date_format is provided, the task description remains unchanged.
    """
    agent = Agent(
        role="test_agent",
        goal="test_goal",
        backstory="test_backstory",
        inject_date=True,
        date_format="invalid",
    )
    
    task = Task(
        description="Test task",
        expected_output="Test output",
        agent=agent,
    )
    
    original_description = task.description
    
    agent.execute_task(task)
    
    assert task.description == original_description
