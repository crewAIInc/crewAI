from __future__ import annotations

import os
import uuid

import pytest
import pytest_asyncio

from a2a.client import ClientFactory
from a2a.types import AgentCard, Message, Part, Role, TaskState, TextPart

from crewai.a2a.updates.polling.handler import PollingHandler
from crewai.a2a.updates.streaming.handler import StreamingHandler


A2A_TEST_ENDPOINT = os.getenv("A2A_TEST_ENDPOINT", "http://localhost:9999")


@pytest_asyncio.fixture
async def a2a_client():
    """Create A2A client for test server."""
    client = await ClientFactory.connect(A2A_TEST_ENDPOINT)
    yield client
    await client.close()


@pytest.fixture
def test_message() -> Message:
    """Create a simple test message."""
    return Message(
        role=Role.user,
        parts=[Part(root=TextPart(text="What is 2 + 2?"))],
        message_id=str(uuid.uuid4()),
    )


@pytest_asyncio.fixture
async def agent_card(a2a_client) -> AgentCard:
    """Fetch the real agent card from the server."""
    return await a2a_client.get_card()


class TestA2AAgentCardFetching:
    """Integration tests for agent card fetching."""

    @pytest.mark.vcr()
    @pytest.mark.asyncio
    async def test_fetch_agent_card(self, a2a_client) -> None:
        """Test fetching an agent card from the server."""
        card = await a2a_client.get_card()

        assert card is not None
        assert card.name == "GPT Assistant"
        assert card.url is not None
        assert card.capabilities is not None
        assert card.capabilities.streaming is True


class TestA2APollingIntegration:
    """Integration tests for A2A polling handler."""

    @pytest.mark.vcr()
    @pytest.mark.asyncio
    async def test_polling_completes_task(
        self,
        a2a_client,
        test_message: Message,
        agent_card: AgentCard,
    ) -> None:
        """Test that polling handler completes a task successfully."""
        new_messages: list[Message] = []

        result = await PollingHandler.execute(
            client=a2a_client,
            message=test_message,
            new_messages=new_messages,
            agent_card=agent_card,
            polling_interval=0.5,
            polling_timeout=30.0,
        )

        assert isinstance(result, dict)
        assert result["status"] == TaskState.completed
        assert result.get("result") is not None
        assert "4" in result["result"]


class TestA2AStreamingIntegration:
    """Integration tests for A2A streaming handler."""

    @pytest.mark.vcr()
    @pytest.mark.asyncio
    async def test_streaming_completes_task(
        self,
        a2a_client,
        test_message: Message,
        agent_card: AgentCard,
    ) -> None:
        """Test that streaming handler completes a task successfully."""
        new_messages: list[Message] = []

        result = await StreamingHandler.execute(
            client=a2a_client,
            message=test_message,
            new_messages=new_messages,
            agent_card=agent_card,
            endpoint=agent_card.url,
        )

        assert isinstance(result, dict)
        assert result["status"] == TaskState.completed
        assert result.get("result") is not None


class TestA2ATaskOperations:
    """Integration tests for task operations."""

    @pytest.mark.vcr()
    @pytest.mark.asyncio
    async def test_send_message_and_get_response(
        self,
        a2a_client,
        test_message: Message,
    ) -> None:
        """Test sending a message and getting a response."""
        from a2a.types import Task

        final_task: Task | None = None
        async for event in a2a_client.send_message(test_message):
            if isinstance(event, tuple) and len(event) >= 1:
                task, _ = event
                if isinstance(task, Task):
                    final_task = task

        assert final_task is not None
        assert final_task.id is not None
        assert final_task.status is not None
        assert final_task.status.state == TaskState.completed


class TestA2APushNotificationHandler:
    """Tests for push notification handler.

    These tests use mocks for the result store since webhook callbacks
    are incoming requests that can't be recorded with VCR.
    """

    @pytest.fixture
    def mock_agent_card(self) -> AgentCard:
        """Create a minimal valid agent card for testing."""
        from a2a.types import AgentCapabilities

        return AgentCard(
            name="Test Agent",
            description="Test agent for push notification tests",
            url="http://localhost:9999",
            version="1.0.0",
            capabilities=AgentCapabilities(streaming=True, push_notifications=True),
            default_input_modes=["text"],
            default_output_modes=["text"],
            skills=[],
        )

    @pytest.fixture
    def mock_task(self) -> "Task":
        """Create a minimal valid task for testing."""
        from a2a.types import Task, TaskStatus

        return Task(
            id="task-123",
            context_id="ctx-123",
            status=TaskStatus(state=TaskState.working),
        )

    @pytest.mark.asyncio
    async def test_push_handler_waits_for_result(
        self,
        mock_agent_card: AgentCard,
        mock_task,
    ) -> None:
        """Test that push handler waits for result from store."""
        from unittest.mock import AsyncMock, MagicMock

        from a2a.types import Task, TaskStatus
        from pydantic import AnyHttpUrl

        from crewai.a2a.updates.push_notifications.config import PushNotificationConfig
        from crewai.a2a.updates.push_notifications.handler import PushNotificationHandler

        completed_task = Task(
            id="task-123",
            context_id="ctx-123",
            status=TaskStatus(state=TaskState.completed),
            history=[],
        )

        mock_store = MagicMock()
        mock_store.wait_for_result = AsyncMock(return_value=completed_task)

        async def mock_send_message(*args, **kwargs):
            yield (mock_task, None)

        mock_client = MagicMock()
        mock_client.send_message = mock_send_message

        config = PushNotificationConfig(
            url=AnyHttpUrl("http://localhost:8080/a2a/callback"),
            token="secret-token",
            result_store=mock_store,
        )

        test_msg = Message(
            role=Role.user,
            parts=[Part(root=TextPart(text="What is 2+2?"))],
            message_id="msg-001",
        )

        new_messages: list[Message] = []

        result = await PushNotificationHandler.execute(
            client=mock_client,
            message=test_msg,
            new_messages=new_messages,
            agent_card=mock_agent_card,
            config=config,
            result_store=mock_store,
            polling_timeout=30.0,
            polling_interval=1.0,
            endpoint=mock_agent_card.url,
        )

        mock_store.wait_for_result.assert_called_once_with(
            task_id="task-123",
            timeout=30.0,
            poll_interval=1.0,
        )

        assert result["status"] == TaskState.completed

    @pytest.mark.asyncio
    async def test_push_handler_returns_failure_on_timeout(
        self,
        mock_agent_card: AgentCard,
    ) -> None:
        """Test that push handler returns failure when result store times out."""
        from unittest.mock import AsyncMock, MagicMock

        from a2a.types import Task, TaskStatus
        from pydantic import AnyHttpUrl

        from crewai.a2a.updates.push_notifications.config import PushNotificationConfig
        from crewai.a2a.updates.push_notifications.handler import PushNotificationHandler

        mock_store = MagicMock()
        mock_store.wait_for_result = AsyncMock(return_value=None)

        working_task = Task(
            id="task-456",
            context_id="ctx-456",
            status=TaskStatus(state=TaskState.working),
        )

        async def mock_send_message(*args, **kwargs):
            yield (working_task, None)

        mock_client = MagicMock()
        mock_client.send_message = mock_send_message

        config = PushNotificationConfig(
            url=AnyHttpUrl("http://localhost:8080/a2a/callback"),
            token="token",
            result_store=mock_store,
        )

        test_msg = Message(
            role=Role.user,
            parts=[Part(root=TextPart(text="test"))],
            message_id="msg-002",
        )

        new_messages: list[Message] = []

        result = await PushNotificationHandler.execute(
            client=mock_client,
            message=test_msg,
            new_messages=new_messages,
            agent_card=mock_agent_card,
            config=config,
            result_store=mock_store,
            polling_timeout=5.0,
            polling_interval=0.5,
            endpoint=mock_agent_card.url,
        )

        assert result["status"] == TaskState.failed
        assert "timeout" in result.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_push_handler_requires_config(
        self,
        mock_agent_card: AgentCard,
    ) -> None:
        """Test that push handler fails gracefully without config."""
        from unittest.mock import MagicMock

        from crewai.a2a.updates.push_notifications.handler import PushNotificationHandler

        mock_client = MagicMock()

        test_msg = Message(
            role=Role.user,
            parts=[Part(root=TextPart(text="test"))],
            message_id="msg-003",
        )

        new_messages: list[Message] = []

        result = await PushNotificationHandler.execute(
            client=mock_client,
            message=test_msg,
            new_messages=new_messages,
            agent_card=mock_agent_card,
            endpoint=mock_agent_card.url,
        )

        assert result["status"] == TaskState.failed
        assert "config" in result.get("error", "").lower()
