"""Tests for A2A delegation tool behavior, including trust_remote_completion_status."""

from unittest.mock import MagicMock, patch

import pytest

from crewai.a2a.config import A2AClientConfig, A2AConfig


try:
    from a2a.types import TaskState  # noqa: F401

    A2A_SDK_INSTALLED = True
except ImportError:
    A2A_SDK_INSTALLED = False


def _create_mock_agent_card(name: str = "Test", url: str = "http://test-endpoint.com/"):
    """Create a mock agent card with the attributes A2ADelegationTool reads."""
    mock_card = MagicMock()
    mock_card.name = name
    mock_card.url = url
    mock_card.description = "A test agent"
    mock_card.skills = []
    mock_card.model_dump.return_value = {"name": name, "url": url}
    return mock_card


@pytest.mark.skipif(not A2A_SDK_INSTALLED, reason="Requires a2a-sdk to be installed")
def test_delegation_tool_returns_remote_result_on_completion():
    """A successful remote completion is returned to the local LLM as the tool result."""
    from a2a.types import TaskState

    from crewai import Agent, Task
    from crewai.a2a.tools import A2ADelegationState, build_a2a_tools

    config = A2AClientConfig(endpoint="http://test-endpoint.com")
    agent = Agent(role="manager", goal="coordinate", backstory="test", a2a=config)
    task = Task(description="test", expected_output="test", agent=agent)

    card = _create_mock_agent_card()
    state = A2ADelegationState(agent=agent, task=task)
    tools = build_a2a_tools([config], {config.endpoint: card}, state)
    assert len(tools) == 1
    tool = tools[0]

    with patch("crewai.a2a.tools.execute_a2a_delegation") as mock_execute:
        mock_execute.return_value = {
            "status": TaskState.completed,
            "result": "Done by remote",
            "history": [],
        }
        result = tool._run(message="Please help")

    assert result == "Done by remote"
    assert mock_execute.call_count == 1


@pytest.mark.skipif(not A2A_SDK_INSTALLED, reason="Requires a2a-sdk to be installed")
def test_delegation_tool_records_completed_task_in_references():
    """When a remote task completes with a task_id, it goes into reference_task_ids."""
    from a2a.types import TaskState

    from crewai import Agent, Task
    from crewai.a2a.tools import A2ADelegationState, build_a2a_tools

    config = A2AClientConfig(endpoint="http://test-endpoint.com")
    agent = Agent(role="manager", goal="coordinate", backstory="test", a2a=config)
    task = Task(description="test", expected_output="test", agent=agent)

    card = _create_mock_agent_card()
    state = A2ADelegationState(agent=agent, task=task)
    [tool] = build_a2a_tools([config], {config.endpoint: card}, state)

    history_msg = MagicMock()
    history_msg.task_id = "remote-task-1"
    history_msg.context_id = "ctx-1"

    with patch("crewai.a2a.tools.execute_a2a_delegation") as mock_execute:
        mock_execute.return_value = {
            "status": TaskState.completed,
            "result": "Done",
            "history": [history_msg],
        }
        tool._run(message="Please help")

    endpoint_state = state._per_endpoint[config.endpoint]
    assert "remote-task-1" in endpoint_state.reference_task_ids
    assert endpoint_state.task_id is None
    assert task.config is not None
    assert task.config["reference_task_ids"] == ["remote-task-1"]


@pytest.mark.skipif(not A2A_SDK_INSTALLED, reason="Requires a2a-sdk to be installed")
def test_delegation_tool_returns_error_message_on_failure():
    """A non-completed/non-input-required status surfaces as a readable error string."""
    from a2a.types import TaskState

    from crewai import Agent, Task
    from crewai.a2a.tools import A2ADelegationState, build_a2a_tools

    config = A2AClientConfig(endpoint="http://test-endpoint.com")
    agent = Agent(role="manager", goal="coordinate", backstory="test", a2a=config)
    task = Task(description="test", expected_output="test", agent=agent)

    card = _create_mock_agent_card()
    state = A2ADelegationState(agent=agent, task=task)
    [tool] = build_a2a_tools([config], {config.endpoint: card}, state)

    with patch("crewai.a2a.tools.execute_a2a_delegation") as mock_execute:
        mock_execute.return_value = {
            "status": TaskState.failed,
            "error": "remote agent unreachable",
            "history": [],
        }
        result = tool._run(message="Please help")

    assert "remote agent unreachable" in result


@pytest.mark.skipif(not A2A_SDK_INSTALLED, reason="Requires a2a-sdk to be installed")
def test_delegation_tool_respects_max_turns_via_usage_count():
    """A2AConfig.max_turns wires through to BaseTool.max_usage_count."""
    from crewai import Agent, Task
    from crewai.a2a.tools import A2ADelegationState, build_a2a_tools

    config = A2AClientConfig(endpoint="http://test-endpoint.com", max_turns=2)
    agent = Agent(role="manager", goal="coordinate", backstory="test", a2a=config)
    task = Task(description="test", expected_output="test", agent=agent)

    card = _create_mock_agent_card()
    state = A2ADelegationState(agent=agent, task=task)
    [tool] = build_a2a_tools([config], {config.endpoint: card}, state)

    assert tool.max_usage_count == 2


def test_default_trust_remote_completion_status_is_false():
    """Verify that default value of trust_remote_completion_status is False."""
    a2a_config = A2AConfig(endpoint="http://test-endpoint.com")
    assert a2a_config.trust_remote_completion_status is False