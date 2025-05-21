from datetime import datetime
from unittest.mock import patch

from crewai.agent import Agent
from crewai.task import Task


def test_agent_inject_date():
    """Test that the inject_date flag injects the current date into the task.
    
    Tests that when inject_date=True, the current date is added to the task description.
    """
    with patch('datetime.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2025, 1, 1)
        
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
        
        # Store original description
        original_description = task.description
        
        agent._inject_date_to_task(task)
        
        assert "Current Date: 2025-01-01" in task.description
        assert task.description != original_description


def test_agent_without_inject_date():
    """Test that without inject_date flag, no date is injected.
    
    Tests that when inject_date=False (default), no date is added to the task description.
    """
    agent = Agent(
        role="test_agent",
        goal="test_goal",
        backstory="test_backstory",
        # inject_date is False by default
    )
    
    task = Task(
        description="Test task",
        expected_output="Test output",
        agent=agent,
    )
    
    original_description = task.description
    
    agent._inject_date_to_task(task)
    
    assert task.description == original_description


def test_agent_inject_date_custom_format():
    """Test that the inject_date flag with custom date_format works correctly.
    
    Tests that when inject_date=True with a custom date_format, the date is formatted correctly.
    """
    with patch('datetime.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2025, 1, 1)
        
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
        
        # Store original description
        original_description = task.description
        
        agent._inject_date_to_task(task)
        
        assert "Current Date: 01/01/2025" in task.description
        assert task.description != original_description


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
    
    agent._inject_date_to_task(task)
    
    assert task.description == original_description
