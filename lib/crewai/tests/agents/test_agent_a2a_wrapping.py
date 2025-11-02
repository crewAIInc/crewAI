"""Test A2A wrapper is only applied when a2a is passed to Agent."""

from unittest.mock import patch

import pytest

from crewai import Agent
from crewai.a2a.config import A2AConfig

try:
    import a2a  # noqa: F401

    A2A_SDK_INSTALLED = True
except ImportError:
    A2A_SDK_INSTALLED = False


def test_agent_without_a2a_has_no_wrapper():
    """Verify that agents without a2a don't get the wrapper applied."""
    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
    )

    assert agent.a2a is None
    assert callable(agent.execute_task)


@pytest.mark.skipif(
    True,
    reason="Requires a2a-sdk to be installed. This test verifies wrapper is applied when a2a is set.",
)
def test_agent_with_a2a_has_wrapper():
    """Verify that agents with a2a get the wrapper applied."""
    a2a_config = A2AConfig(
        endpoint="http://test-endpoint.com",
    )

    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        a2a=a2a_config,
    )

    assert agent.a2a is not None
    assert agent.a2a.endpoint == "http://test-endpoint.com"
    assert callable(agent.execute_task)


@pytest.mark.skipif(not A2A_SDK_INSTALLED, reason="Requires a2a-sdk to be installed")
def test_agent_with_a2a_creates_successfully():
    """Verify that creating an agent with a2a succeeds and applies wrapper."""
    a2a_config = A2AConfig(
        endpoint="http://test-endpoint.com",
    )

    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        a2a=a2a_config,
    )

    assert agent.a2a is not None
    assert agent.a2a.endpoint == "http://test-endpoint.com/"
    assert callable(agent.execute_task)
    assert hasattr(agent.execute_task, "__wrapped__")


def test_multiple_agents_without_a2a():
    """Verify that multiple agents without a2a work correctly."""
    agent1 = Agent(
        role="agent 1",
        goal="test goal",
        backstory="test backstory",
    )

    agent2 = Agent(
        role="agent 2",
        goal="test goal",
        backstory="test backstory",
    )

    assert agent1.a2a is None
    assert agent2.a2a is None
    assert callable(agent1.execute_task)
    assert callable(agent2.execute_task)


@pytest.mark.skipif(not A2A_SDK_INSTALLED, reason="Requires a2a-sdk to be installed")
def test_wrapper_is_applied_differently_per_instance():
    """Verify that agents with and without a2a have different execute_task methods."""
    agent_without_a2a = Agent(
        role="agent without a2a",
        goal="test goal",
        backstory="test backstory",
    )

    a2a_config = A2AConfig(endpoint="http://test-endpoint.com")
    agent_with_a2a = Agent(
        role="agent with a2a",
        goal="test goal",
        backstory="test backstory",
        a2a=a2a_config,
    )

    assert agent_without_a2a.execute_task.__func__ is not agent_with_a2a.execute_task.__func__
    assert not hasattr(agent_without_a2a.execute_task, "__wrapped__")
    assert hasattr(agent_with_a2a.execute_task, "__wrapped__")
