"""Tests for the A2A protocol integration."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from crewai.agent import Agent
from crewai.a2a import A2AAgentIntegration, A2AClient, A2AServer, InMemoryTaskManager
from crewai.task import Task
from crewai.types.a2a import (
    Message,
    Task as A2ATask,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)


@pytest.fixture
def agent():
    """Create an agent with A2A enabled."""
    return Agent(
        role="test_agent",
        goal="Test A2A protocol",
        backstory="I am a test agent",
        a2a_enabled=True,
        a2a_url="http://localhost:8000",
    )


@pytest.fixture
def task():
    """Create a task."""
    return Task(
        description="Test task",
    )


@pytest.fixture
def a2a_task():
    """Create an A2A task."""
    return A2ATask(
        id="test_task_id",
        history=[
            Message(
                role="user",
                parts=[TextPart(text="Test task description")],
            )
        ],
    )


@pytest.fixture
def a2a_integration():
    """Create an A2A integration."""
    return A2AAgentIntegration()


@pytest.fixture
def a2a_client():
    """Create an A2A client."""
    return A2AClient(base_url="http://localhost:8000")


@pytest.fixture
def task_manager():
    """Create a task manager."""
    return InMemoryTaskManager()


class TestA2AIntegration:
    """Tests for the A2A protocol integration."""

    def test_agent_a2a_attributes(self, agent):
        """Test that the agent has A2A attributes."""
        assert agent.a2a_enabled is True
        assert agent.a2a_url == "http://localhost:8000"
        assert agent._a2a_integration is not None

    @patch("crewai.a2a.agent.A2AAgentIntegration.execute_task_via_a2a")
    def test_execute_task_via_a2a(self, mock_execute, agent):
        """Test executing a task via A2A."""
        mock_execute.return_value = asyncio.Future()
        mock_execute.return_value.set_result("Task result")

        result = asyncio.run(
            agent.execute_task_via_a2a(
                task_description="Test task",
                context="Test context",
            )
        )

        assert result == "Task result"
        mock_execute.assert_called_once_with(
            agent_url="http://localhost:8000",
            task_description="Test task",
            context="Test context",
            api_key=None,
            timeout=300,
        )

    @patch("crewai.agent.Agent.execute_task")
    def test_handle_a2a_task(self, mock_execute, agent):
        """Test handling an A2A task."""
        mock_execute.return_value = "Task result"

        result = asyncio.run(
            agent.handle_a2a_task(
                task_id="test_task_id",
                task_description="Test task",
                context="Test context",
            )
        )

        assert result == "Task result"
        mock_execute.assert_called_once()
        args, kwargs = mock_execute.call_args
        assert args[0].description == "Test task"
        assert kwargs["context"] == "Test context"

    def test_a2a_disabled(self, agent):
        """Test that A2A methods raise ValueError when A2A is disabled."""
        agent.a2a_enabled = False

        with pytest.raises(ValueError, match="A2A protocol is not enabled for this agent"):
            asyncio.run(
                agent.execute_task_via_a2a(
                    task_description="Test task",
                )
            )

        with pytest.raises(ValueError, match="A2A protocol is not enabled for this agent"):
            asyncio.run(
                agent.handle_a2a_task(
                    task_id="test_task_id",
                    task_description="Test task",
                )
            )

    def test_no_agent_url(self, agent):
        """Test that execute_task_via_a2a raises ValueError when no agent URL is provided."""
        agent.a2a_url = None

        with pytest.raises(ValueError, match="No A2A agent URL provided"):
            asyncio.run(
                agent.execute_task_via_a2a(
                    task_description="Test task",
                )
            )


class TestA2AAgentIntegration:
    """Tests for the A2AAgentIntegration class."""

    @patch("crewai.a2a.client.A2AClient.send_task_streaming")
    async def test_execute_task_via_a2a(self, mock_send_task, a2a_integration):
        """Test executing a task via A2A."""
        queue = asyncio.Queue()
        await queue.put(
            TaskStatusUpdateEvent(
                task_id="test_task_id",
                status=TaskStatus(
                    state=TaskState.COMPLETED,
                    message=Message(
                        role="agent",
                        parts=[TextPart(text="Task result")],
                    ),
                ),
                final=True,
            )
        )

        mock_send_task.return_value = queue

        result = await a2a_integration.execute_task_via_a2a(
            agent_url="http://localhost:8000",
            task_description="Test task",
            context="Test context",
        )

        assert result == "Task result"
        mock_send_task.assert_called_once()


class TestA2AServer:
    """Tests for the A2AServer class."""

    @patch("fastapi.FastAPI.post")
    def test_server_initialization(self, mock_post, task_manager):
        """Test server initialization."""
        server = A2AServer(task_manager=task_manager)
        assert server.task_manager == task_manager
        assert server.app is not None
        assert mock_post.call_count == 4  # 4 endpoints registered


class TestA2AClient:
    """Tests for the A2AClient class."""

    @patch("aiohttp.ClientSession.post")
    async def test_send_task(self, mock_post, a2a_client):
        """Test sending a task."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "id": "test_task_id",
                "history": [
                    {
                        "role": "user",
                        "parts": [{"text": "Test task description"}],
                    }
                ],
            }
        )
        mock_post.return_value.__aenter__.return_value = mock_response

        task = await a2a_client.send_task(
            task_id="test_task_id",
            message=Message(
                role="user",
                parts=[TextPart(text="Test task description")],
            ),
        )

        assert task.id == "test_task_id"
        assert task.history[0].role == "user"
        assert task.history[0].parts[0].text == "Test task description"
        mock_post.assert_called_once()
