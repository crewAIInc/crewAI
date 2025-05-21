import re
from datetime import datetime
from unittest.mock import patch, MagicMock

from crewai.agent import Agent
from crewai.task import Task


def test_agent_inject_date():
    """Test that the inject_date flag injects the current date into the task."""
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
    """Test that without inject_date flag, no date is injected."""
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
