"""Test template handling in Agent creation."""

import pytest

from crewai import Agent, Crew, Task


def test_agent_with_only_system_template():
    """Test that an agent with only system_template works without errors."""
    agent = Agent(
        role="Test Role",
        goal="Test Goal",
        backstory="Test Backstory",
        allow_delegation=False,
        system_template="You are a test agent...",
        # prompt_template is intentionally missing
    )
    
    task = Task(description="Test task", agent=agent, expected_output="Test output")
    
    # This should not raise an error
    try:
        agent.execute_task(task)
        assert True
    except AttributeError:
        pytest.fail("AttributeError was raised with only system_template")


def test_agent_with_missing_response_template():
    """Test that an agent with system_template and prompt_template but no response_template works without errors."""
    agent = Agent(
        role="Test Role",
        goal="Test Goal",
        backstory="Test Backstory",
        allow_delegation=False,
        system_template="You are a test agent...",
        prompt_template="This is a test prompt...",
        # response_template is intentionally missing
    )
    
    task = Task(description="Test task", agent=agent, expected_output="Test output")
    
    # This should not raise an error
    try:
        agent.execute_task(task)
        assert True
    except AttributeError:
        pytest.fail("AttributeError was raised with missing response_template")
