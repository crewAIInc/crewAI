"""Test A2A async execution support.

Tests that verify async execution works correctly without creating new event loops.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crewai import Agent
from crewai.a2a.config import A2AConfig

try:
    from a2a.types import Message, Role

    A2A_SDK_INSTALLED = True
except ImportError:
    A2A_SDK_INSTALLED = False


@pytest.mark.skipif(not A2A_SDK_INSTALLED, reason="Requires a2a-sdk to be installed")
def test_agent_with_a2a_has_async_wrapper():
    """Verify that agents with a2a get the async wrapper applied to aexecute_task."""
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
    assert callable(agent.aexecute_task)
    assert hasattr(agent.aexecute_task, "__wrapped__")


@pytest.mark.skipif(not A2A_SDK_INSTALLED, reason="Requires a2a-sdk to be installed")
def test_async_wrapper_is_applied_differently_per_instance():
    """Verify that agents with and without a2a have different aexecute_task methods."""
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

    assert (
        agent_without_a2a.aexecute_task.__func__
        is not agent_with_a2a.aexecute_task.__func__
    )
    assert not hasattr(agent_without_a2a.aexecute_task, "__wrapped__")
    assert hasattr(agent_with_a2a.aexecute_task, "__wrapped__")


@pytest.mark.skipif(not A2A_SDK_INSTALLED, reason="Requires a2a-sdk to be installed")
@pytest.mark.asyncio
async def test_async_delegate_to_a2a_does_not_create_new_event_loop():
    """Verify that async A2A delegation doesn't create a new event loop."""
    from crewai.a2a.wrapper import _adelegate_to_a2a
    from crewai import Task

    a2a_config = A2AConfig(
        endpoint="http://test-endpoint.com",
        trust_remote_completion_status=True,
    )

    agent = Agent(
        role="test manager",
        goal="coordinate",
        backstory="test",
        a2a=a2a_config,
    )

    task = Task(description="test", expected_output="test", agent=agent)

    class MockResponse:
        is_a2a = True
        message = "Please help"
        a2a_ids = ["http://test-endpoint.com/"]

    async def mock_original_fn(self, task, context, tools):
        return '{"is_a2a": false, "message": "Done", "a2a_ids": []}'

    with (
        patch("crewai.a2a.wrapper.aexecute_a2a_delegation") as mock_execute,
        patch("crewai.a2a.wrapper._afetch_agent_cards_concurrently") as mock_fetch,
        patch("asyncio.new_event_loop") as mock_new_loop,
    ):
        mock_card = MagicMock()
        mock_card.name = "Test"
        mock_fetch.return_value = ({"http://test-endpoint.com/": mock_card}, {})

        mock_execute.return_value = {
            "status": "completed",
            "result": "Done by remote",
            "history": [],
        }

        result = await _adelegate_to_a2a(
            self=agent,
            agent_response=MockResponse(),
            task=task,
            original_fn=mock_original_fn,
            context=None,
            tools=None,
            agent_cards={"http://test-endpoint.com/": mock_card},
            original_task_description="test",
        )

        assert result == "Done by remote"
        mock_new_loop.assert_not_called()


@pytest.mark.skipif(not A2A_SDK_INSTALLED, reason="Requires a2a-sdk to be installed")
@pytest.mark.asyncio
async def test_aexecute_a2a_delegation_does_not_create_new_event_loop():
    """Verify that aexecute_a2a_delegation doesn't create a new event loop."""
    from crewai.a2a.utils import aexecute_a2a_delegation

    with (
        patch(
            "crewai.a2a.utils._execute_a2a_delegation_async"
        ) as mock_execute_async,
        patch("asyncio.new_event_loop") as mock_new_loop,
    ):
        mock_execute_async.return_value = {
            "status": "completed",
            "result": "Done",
            "history": [],
        }

        result = await aexecute_a2a_delegation(
            endpoint="http://test-endpoint.com",
            auth=None,
            timeout=30,
            task_description="test task",
            agent_id="test-agent",
        )

        assert result["status"] == "completed"
        mock_new_loop.assert_not_called()
        mock_execute_async.assert_called_once()


@pytest.mark.skipif(not A2A_SDK_INSTALLED, reason="Requires a2a-sdk to be installed")
@pytest.mark.asyncio
async def test_afetch_agent_card_does_not_create_new_event_loop():
    """Verify that afetch_agent_card doesn't create a new event loop."""
    from crewai.a2a.utils import afetch_agent_card

    with (
        patch("crewai.a2a.utils._fetch_agent_card_async") as mock_fetch_async,
        patch("asyncio.new_event_loop") as mock_new_loop,
    ):
        mock_card = MagicMock()
        mock_card.name = "Test Agent"
        mock_fetch_async.return_value = mock_card

        result = await afetch_agent_card(
            endpoint="http://test-endpoint.com",
            auth=None,
            timeout=30,
            use_cache=False,
        )

        assert result.name == "Test Agent"
        mock_new_loop.assert_not_called()
        mock_fetch_async.assert_called_once()


@pytest.mark.skipif(not A2A_SDK_INSTALLED, reason="Requires a2a-sdk to be installed")
@pytest.mark.asyncio
async def test_afetch_agent_cards_concurrently():
    """Verify that _afetch_agent_cards_concurrently fetches cards using asyncio.gather."""
    from crewai.a2a.wrapper import _afetch_agent_cards_concurrently

    a2a_configs = [
        A2AConfig(endpoint="http://test-endpoint-1.com"),
        A2AConfig(endpoint="http://test-endpoint-2.com"),
    ]

    with patch("crewai.a2a.wrapper.afetch_agent_card") as mock_fetch:
        mock_card1 = MagicMock()
        mock_card1.name = "Agent 1"
        mock_card2 = MagicMock()
        mock_card2.name = "Agent 2"

        async def side_effect(endpoint, auth, timeout):
            if "endpoint-1" in endpoint:
                return mock_card1
            return mock_card2

        mock_fetch.side_effect = side_effect

        agent_cards, failed_agents = await _afetch_agent_cards_concurrently(a2a_configs)

        assert len(agent_cards) == 2
        assert len(failed_agents) == 0
        assert mock_fetch.call_count == 2


@pytest.mark.skipif(not A2A_SDK_INSTALLED, reason="Requires a2a-sdk to be installed")
@pytest.mark.asyncio
async def test_aexecute_task_with_a2a_uses_async_path():
    """Verify that _aexecute_task_with_a2a uses the async delegation path."""
    from crewai.a2a.wrapper import _aexecute_task_with_a2a
    from crewai.a2a.utils import get_a2a_agents_and_response_model
    from crewai import Task

    a2a_config = A2AConfig(
        endpoint="http://test-endpoint.com",
    )

    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        a2a=a2a_config,
    )

    task = Task(description="test task", expected_output="test output", agent=agent)

    a2a_agents, agent_response_model = get_a2a_agents_and_response_model(a2a_config)

    async def mock_original_fn(self, task, context, tools):
        return '{"is_a2a": false, "message": "Direct response", "a2a_ids": []}'

    with (
        patch("crewai.a2a.wrapper._afetch_agent_cards_concurrently") as mock_fetch,
    ):
        mock_card = MagicMock()
        mock_card.name = "Test"
        mock_fetch.return_value = ({"http://test-endpoint.com/": mock_card}, {})

        from crewai.a2a.extensions.base import ExtensionRegistry

        result = await _aexecute_task_with_a2a(
            self=agent,
            a2a_agents=a2a_agents,
            original_fn=mock_original_fn,
            task=task,
            agent_response_model=agent_response_model,
            context=None,
            tools=None,
            extension_registry=ExtensionRegistry(),
        )

        assert result == "Direct response"
        mock_fetch.assert_called_once()


@pytest.mark.skipif(not A2A_SDK_INSTALLED, reason="Requires a2a-sdk to be installed")
@pytest.mark.asyncio
async def test_async_execution_in_running_event_loop():
    """Verify that async A2A execution works correctly within a running event loop.

    This test simulates the scenario described in issue #4162 where A2A is called
    from an async context that already has a running event loop.
    """
    from crewai.a2a.utils import aexecute_a2a_delegation

    current_loop = asyncio.get_running_loop()
    assert current_loop is not None

    with patch(
        "crewai.a2a.utils._execute_a2a_delegation_async"
    ) as mock_execute_async:
        mock_execute_async.return_value = {
            "status": "completed",
            "result": "Success from async context",
            "history": [],
        }

        result = await aexecute_a2a_delegation(
            endpoint="http://test-endpoint.com",
            auth=None,
            timeout=30,
            task_description="test task from async context",
            agent_id="test-agent",
        )

        assert result["status"] == "completed"
        assert result["result"] == "Success from async context"
