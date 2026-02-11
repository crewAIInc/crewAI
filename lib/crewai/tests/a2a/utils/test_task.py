"""Tests for A2A task utilities."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.types import Message, Task as A2ATask, TaskState, TaskStatus

from crewai.a2a.utils.task import cancel, cancellable, execute


@pytest.fixture
def mock_agent() -> MagicMock:
    """Create a mock CrewAI agent."""
    agent = MagicMock()
    agent.role = "Test Agent"
    agent.tools = []
    agent.aexecute_task = AsyncMock(return_value="Task completed successfully")
    return agent


@pytest.fixture
def mock_task(mock_context: MagicMock) -> MagicMock:
    """Create a mock Task."""
    task = MagicMock()
    task.id = mock_context.task_id
    task.name = "Mock Task"
    task.description = "Mock task description"
    return task


@pytest.fixture
def mock_context() -> MagicMock:
    """Create a mock RequestContext."""
    context = MagicMock(spec=RequestContext)
    context.task_id = "test-task-123"
    context.context_id = "test-context-456"
    context.get_user_input.return_value = "Test user message"
    context.message = MagicMock(spec=Message)
    context.message.parts = []
    context.current_task = None
    return context


@pytest.fixture
def mock_event_queue() -> AsyncMock:
    """Create a mock EventQueue."""
    queue = AsyncMock(spec=EventQueue)
    queue.enqueue_event = AsyncMock()
    return queue


@pytest_asyncio.fixture(autouse=True)
async def clear_cache(mock_context: MagicMock) -> None:
    """Clear cancel flag from cache before each test."""
    from aiocache import caches

    cache = caches.get("default")
    await cache.delete(f"cancel:{mock_context.task_id}")


class TestCancellableDecorator:
    """Tests for the cancellable decorator."""

    @pytest.mark.asyncio
    async def test_executes_function_without_context(self) -> None:
        """Function executes normally when no RequestContext is provided."""
        call_count = 0

        @cancellable
        async def my_func(value: int) -> int:
            nonlocal call_count
            call_count += 1
            return value * 2

        result = await my_func(5)

        assert result == 10
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_executes_function_with_context(self, mock_context: MagicMock) -> None:
        """Function executes normally with RequestContext when not cancelled."""
        @cancellable
        async def my_func(context: RequestContext) -> str:
            await asyncio.sleep(0.01)
            return "completed"

        result = await my_func(mock_context)

        assert result == "completed"

    @pytest.mark.asyncio
    async def test_cancellation_raises_cancelled_error(
        self, mock_context: MagicMock
    ) -> None:
        """Function raises CancelledError when cancel flag is set."""
        from aiocache import caches

        cache = caches.get("default")

        @cancellable
        async def slow_func(context: RequestContext) -> str:
            await asyncio.sleep(1.0)
            return "should not reach"

        await cache.set(f"cancel:{mock_context.task_id}", True)

        with pytest.raises(asyncio.CancelledError):
            await slow_func(mock_context)

    @pytest.mark.asyncio
    async def test_cleanup_removes_cancel_flag(self, mock_context: MagicMock) -> None:
        """Cancel flag is cleaned up after execution."""
        from aiocache import caches

        cache = caches.get("default")

        @cancellable
        async def quick_func(context: RequestContext) -> str:
            return "done"

        await quick_func(mock_context)

        flag = await cache.get(f"cancel:{mock_context.task_id}")
        assert flag is None

    @pytest.mark.asyncio
    async def test_extracts_context_from_kwargs(self, mock_context: MagicMock) -> None:
        """Context can be passed as keyword argument."""
        @cancellable
        async def my_func(value: int, context: RequestContext | None = None) -> int:
            return value + 1

        result = await my_func(10, context=mock_context)

        assert result == 11


class TestExecute:
    """Tests for the execute function."""

    @pytest.mark.asyncio
    async def test_successful_execution(
        self,
        mock_agent: MagicMock,
        mock_context: MagicMock,
        mock_event_queue: AsyncMock,
        mock_task: MagicMock,
    ) -> None:
        """Execute completes successfully and enqueues completed task."""
        with (
            patch("crewai.a2a.utils.task.Task", return_value=mock_task),
            patch("crewai.a2a.utils.task.crewai_event_bus") as mock_bus,
        ):
            await execute(mock_agent, mock_context, mock_event_queue)

        mock_agent.aexecute_task.assert_called_once()
        mock_event_queue.enqueue_event.assert_called_once()
        assert mock_bus.emit.call_count == 2

    @pytest.mark.asyncio
    async def test_emits_started_event(
        self,
        mock_agent: MagicMock,
        mock_context: MagicMock,
        mock_event_queue: AsyncMock,
        mock_task: MagicMock,
    ) -> None:
        """Execute emits A2AServerTaskStartedEvent."""
        with (
            patch("crewai.a2a.utils.task.Task", return_value=mock_task),
            patch("crewai.a2a.utils.task.crewai_event_bus") as mock_bus,
        ):
            await execute(mock_agent, mock_context, mock_event_queue)

        first_call = mock_bus.emit.call_args_list[0]
        event = first_call[0][1]

        assert event.type == "a2a_server_task_started"
        assert event.task_id == mock_context.task_id
        assert event.context_id == mock_context.context_id

    @pytest.mark.asyncio
    async def test_emits_completed_event(
        self,
        mock_agent: MagicMock,
        mock_context: MagicMock,
        mock_event_queue: AsyncMock,
        mock_task: MagicMock,
    ) -> None:
        """Execute emits A2AServerTaskCompletedEvent on success."""
        with (
            patch("crewai.a2a.utils.task.Task", return_value=mock_task),
            patch("crewai.a2a.utils.task.crewai_event_bus") as mock_bus,
        ):
            await execute(mock_agent, mock_context, mock_event_queue)

        second_call = mock_bus.emit.call_args_list[1]
        event = second_call[0][1]

        assert event.type == "a2a_server_task_completed"
        assert event.task_id == mock_context.task_id
        assert event.result == "Task completed successfully"

    @pytest.mark.asyncio
    async def test_emits_failed_event_on_exception(
        self,
        mock_agent: MagicMock,
        mock_context: MagicMock,
        mock_event_queue: AsyncMock,
        mock_task: MagicMock,
    ) -> None:
        """Execute emits A2AServerTaskFailedEvent on exception."""
        mock_agent.aexecute_task = AsyncMock(side_effect=ValueError("Test error"))

        with (
            patch("crewai.a2a.utils.task.Task", return_value=mock_task),
            patch("crewai.a2a.utils.task.crewai_event_bus") as mock_bus,
        ):
            with pytest.raises(Exception):
                await execute(mock_agent, mock_context, mock_event_queue)

        failed_call = mock_bus.emit.call_args_list[1]
        event = failed_call[0][1]

        assert event.type == "a2a_server_task_failed"
        assert "Test error" in event.error

    @pytest.mark.asyncio
    async def test_emits_canceled_event_on_cancellation(
        self,
        mock_agent: MagicMock,
        mock_context: MagicMock,
        mock_event_queue: AsyncMock,
        mock_task: MagicMock,
    ) -> None:
        """Execute emits A2AServerTaskCanceledEvent on CancelledError."""
        mock_agent.aexecute_task = AsyncMock(side_effect=asyncio.CancelledError())

        with (
            patch("crewai.a2a.utils.task.Task", return_value=mock_task),
            patch("crewai.a2a.utils.task.crewai_event_bus") as mock_bus,
        ):
            with pytest.raises(asyncio.CancelledError):
                await execute(mock_agent, mock_context, mock_event_queue)

        canceled_call = mock_bus.emit.call_args_list[1]
        event = canceled_call[0][1]

        assert event.type == "a2a_server_task_canceled"
        assert event.task_id == mock_context.task_id


class TestCancel:
    """Tests for the cancel function."""

    @pytest.mark.asyncio
    async def test_sets_cancel_flag_in_cache(
        self,
        mock_context: MagicMock,
        mock_event_queue: AsyncMock,
    ) -> None:
        """Cancel sets the cancel flag in cache."""
        from aiocache import caches

        cache = caches.get("default")

        await cancel(mock_context, mock_event_queue)

        flag = await cache.get(f"cancel:{mock_context.task_id}")
        assert flag is True

    @pytest.mark.asyncio
    async def test_enqueues_task_status_update_event(
        self,
        mock_context: MagicMock,
        mock_event_queue: AsyncMock,
    ) -> None:
        """Cancel enqueues TaskStatusUpdateEvent with canceled state."""
        await cancel(mock_context, mock_event_queue)

        mock_event_queue.enqueue_event.assert_called_once()
        event = mock_event_queue.enqueue_event.call_args[0][0]

        assert event.task_id == mock_context.task_id
        assert event.context_id == mock_context.context_id
        assert event.status.state == TaskState.canceled
        assert event.final is True

    @pytest.mark.asyncio
    async def test_returns_none_when_no_current_task(
        self,
        mock_context: MagicMock,
        mock_event_queue: AsyncMock,
    ) -> None:
        """Cancel returns None when context has no current_task."""
        mock_context.current_task = None

        result = await cancel(mock_context, mock_event_queue)

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_updated_task_when_current_task_exists(
        self,
        mock_context: MagicMock,
        mock_event_queue: AsyncMock,
    ) -> None:
        """Cancel returns updated task when context has current_task."""
        current_task = MagicMock(spec=A2ATask)
        current_task.status = TaskStatus(state=TaskState.working)
        mock_context.current_task = current_task

        result = await cancel(mock_context, mock_event_queue)

        assert result is current_task
        assert result.status.state == TaskState.canceled

    @pytest.mark.asyncio
    async def test_cleanup_after_cancel(
        self,
        mock_context: MagicMock,
        mock_event_queue: AsyncMock,
    ) -> None:
        """Cancel flag persists for cancellable decorator to detect."""
        from aiocache import caches

        cache = caches.get("default")

        await cancel(mock_context, mock_event_queue)

        flag = await cache.get(f"cancel:{mock_context.task_id}")
        assert flag is True

        await cache.delete(f"cancel:{mock_context.task_id}")


class TestExecuteAndCancelIntegration:
    """Integration tests for execute and cancel working together."""

    @pytest.mark.asyncio
    async def test_cancel_stops_running_execute(
        self,
        mock_agent: MagicMock,
        mock_context: MagicMock,
        mock_event_queue: AsyncMock,
        mock_task: MagicMock,
    ) -> None:
        """Calling cancel stops a running execute."""
        async def slow_task(**kwargs: Any) -> str:
            await asyncio.sleep(2.0)
            return "should not complete"

        mock_agent.aexecute_task = slow_task

        with (
            patch("crewai.a2a.utils.task.Task", return_value=mock_task),
            patch("crewai.a2a.utils.task.crewai_event_bus"),
        ):
            execute_task = asyncio.create_task(
                execute(mock_agent, mock_context, mock_event_queue)
            )

            await asyncio.sleep(0.1)
            await cancel(mock_context, mock_event_queue)

            with pytest.raises(asyncio.CancelledError):
                await execute_task