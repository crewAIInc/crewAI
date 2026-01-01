"""Test A2A delegation to multiple endpoints sequentially.

This test file covers the bug fix for issue #4166 where delegating to a second
A2A agent fails because the task_id from the first agent is in "completed" state.
"""

from unittest.mock import MagicMock, patch

import pytest

from crewai.a2a.config import A2AConfig

try:
    from a2a.types import Message, Part, Role, TextPart

    A2A_SDK_INSTALLED = True
except ImportError:
    A2A_SDK_INSTALLED = False


@pytest.mark.skipif(not A2A_SDK_INSTALLED, reason="Requires a2a-sdk to be installed")
def test_sequential_delegation_to_multiple_endpoints_uses_separate_task_ids():
    """When delegating to multiple A2A endpoints sequentially, each should get a unique task_id.

    This test verifies the fix for issue #4166 where the second A2A delegation
    fails with 'Task is in terminal state: completed' because the task_id from
    the first delegation was being reused.
    """
    from crewai.a2a.wrapper import _delegate_to_a2a
    from crewai import Agent, Task

    # Configure agent with two A2A endpoints
    a2a_configs = [
        A2AConfig(
            endpoint="http://endpoint-a.com",
            trust_remote_completion_status=True,
        ),
        A2AConfig(
            endpoint="http://endpoint-b.com",
            trust_remote_completion_status=True,
        ),
    ]

    agent = Agent(
        role="test manager",
        goal="coordinate",
        backstory="test",
        a2a=a2a_configs,
    )

    task = Task(description="test", expected_output="test", agent=agent)

    # First delegation to endpoint A
    class MockResponseA:
        is_a2a = True
        message = "Please help with task A"
        a2a_ids = ["http://endpoint-a.com/"]

    # Second delegation to endpoint B
    class MockResponseB:
        is_a2a = True
        message = "Please help with task B"
        a2a_ids = ["http://endpoint-b.com/"]

    task_ids_used = []

    def mock_execute_a2a_delegation(**kwargs):
        """Track the task_id used for each delegation."""
        task_ids_used.append(kwargs.get("task_id"))
        endpoint = kwargs.get("endpoint")

        # Create a mock message with a task_id
        mock_message = MagicMock()
        mock_message.task_id = f"task-id-for-{endpoint}"
        mock_message.context_id = None

        return {
            "status": "completed",
            "result": f"Done by {endpoint}",
            "history": [mock_message],
        }

    with (
        patch(
            "crewai.a2a.wrapper.execute_a2a_delegation",
            side_effect=mock_execute_a2a_delegation,
        ) as mock_execute,
        patch("crewai.a2a.wrapper._fetch_agent_cards_concurrently") as mock_fetch,
    ):
        mock_card_a = MagicMock()
        mock_card_a.name = "Agent A"
        mock_card_b = MagicMock()
        mock_card_b.name = "Agent B"
        mock_fetch.return_value = (
            {
                "http://endpoint-a.com/": mock_card_a,
                "http://endpoint-b.com/": mock_card_b,
            },
            {},
        )

        # First delegation to endpoint A
        result_a = _delegate_to_a2a(
            self=agent,
            agent_response=MockResponseA(),
            task=task,
            original_fn=lambda *args, **kwargs: "fallback",
            context=None,
            tools=None,
            agent_cards={
                "http://endpoint-a.com/": mock_card_a,
                "http://endpoint-b.com/": mock_card_b,
            },
            original_task_description="test",
        )

        assert result_a == "Done by http://endpoint-a.com/"

        # Second delegation to endpoint B
        result_b = _delegate_to_a2a(
            self=agent,
            agent_response=MockResponseB(),
            task=task,
            original_fn=lambda *args, **kwargs: "fallback",
            context=None,
            tools=None,
            agent_cards={
                "http://endpoint-a.com/": mock_card_a,
                "http://endpoint-b.com/": mock_card_b,
            },
            original_task_description="test",
        )

        assert result_b == "Done by http://endpoint-b.com/"

        # Verify that the second delegation used a different (None) task_id
        # The first call should have task_id=None (no prior task_id for endpoint A)
        # The second call should also have task_id=None (no prior task_id for endpoint B)
        assert len(task_ids_used) == 2
        assert task_ids_used[0] is None  # First delegation to endpoint A
        assert task_ids_used[1] is None  # Second delegation to endpoint B (not reusing A's task_id)


@pytest.mark.skipif(not A2A_SDK_INSTALLED, reason="Requires a2a-sdk to be installed")
def test_completed_task_ids_are_not_persisted_for_reuse():
    """Completed task IDs should NOT be persisted for reuse.

    The A2A protocol rejects task IDs that are in terminal state (completed/failed).
    This test verifies that completed task IDs are not stored in task.config
    for future delegations, so each new delegation gets a fresh task_id.
    """
    from crewai.a2a.wrapper import _delegate_to_a2a
    from crewai import Agent, Task

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

    task_ids_used = []

    def mock_execute_a2a_delegation(**kwargs):
        """Track the task_id used for each call."""
        task_ids_used.append(kwargs.get("task_id"))

        # Create a mock message with a task_id
        mock_message = MagicMock()
        mock_message.task_id = "completed-task-id"
        mock_message.context_id = None

        return {
            "status": "completed",
            "result": "Done",
            "history": [mock_message],
        }

    with (
        patch(
            "crewai.a2a.wrapper.execute_a2a_delegation",
            side_effect=mock_execute_a2a_delegation,
        ),
        patch("crewai.a2a.wrapper._fetch_agent_cards_concurrently") as mock_fetch,
    ):
        mock_card = MagicMock()
        mock_card.name = "Test"
        mock_fetch.return_value = ({"http://test-endpoint.com/": mock_card}, {})

        # First delegation
        _delegate_to_a2a(
            self=agent,
            agent_response=MockResponse(),
            task=task,
            original_fn=lambda *args, **kwargs: "fallback",
            context=None,
            tools=None,
            agent_cards={"http://test-endpoint.com/": mock_card},
            original_task_description="test",
        )

        # Verify that completed task IDs are NOT stored in a2a_task_ids_by_endpoint
        # because they can't be reused (A2A protocol rejects terminal state task IDs)
        if task.config is not None:
            a2a_task_ids = task.config.get("a2a_task_ids_by_endpoint", {})
            # The endpoint should NOT have a stored task_id since it completed
            assert "http://test-endpoint.com/" not in a2a_task_ids

        # Second delegation to the SAME endpoint should also get a fresh task_id
        _delegate_to_a2a(
            self=agent,
            agent_response=MockResponse(),
            task=task,
            original_fn=lambda *args, **kwargs: "fallback",
            context=None,
            tools=None,
            agent_cards={"http://test-endpoint.com/": mock_card},
            original_task_description="test",
        )

        # Verify that BOTH calls used None as task_id (fresh task for each)
        # because completed task IDs are not persisted
        assert len(task_ids_used) == 2
        assert task_ids_used[0] is None  # First call - new conversation
        assert task_ids_used[1] is None  # Second call - also new (completed IDs not reused)


@pytest.mark.skipif(not A2A_SDK_INSTALLED, reason="Requires a2a-sdk to be installed")
def test_reference_task_ids_are_tracked_for_completed_tasks():
    """Completed task IDs should be added to reference_task_ids for tracking.

    While completed task IDs can't be reused for new delegations, they should
    still be tracked in reference_task_ids for context/history purposes.
    """
    from crewai.a2a.wrapper import _delegate_to_a2a
    from crewai import Agent, Task

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

    def mock_execute_a2a_delegation(**kwargs):
        mock_message = MagicMock()
        mock_message.task_id = "unique-task-id-123"
        mock_message.context_id = None

        return {
            "status": "completed",
            "result": "Done",
            "history": [mock_message],
        }

    with (
        patch(
            "crewai.a2a.wrapper.execute_a2a_delegation",
            side_effect=mock_execute_a2a_delegation,
        ),
        patch("crewai.a2a.wrapper._fetch_agent_cards_concurrently") as mock_fetch,
    ):
        mock_card = MagicMock()
        mock_card.name = "Test"
        mock_fetch.return_value = ({"http://test-endpoint.com/": mock_card}, {})

        _delegate_to_a2a(
            self=agent,
            agent_response=MockResponse(),
            task=task,
            original_fn=lambda *args, **kwargs: "fallback",
            context=None,
            tools=None,
            agent_cards={"http://test-endpoint.com/": mock_card},
            original_task_description="test",
        )

        # Verify the completed task_id is tracked in reference_task_ids
        assert task.config is not None
        assert "reference_task_ids" in task.config
        assert "unique-task-id-123" in task.config["reference_task_ids"]
