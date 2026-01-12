"""Tests for EventListener execution_spans cleanup to prevent memory leaks."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from crewai.events.event_bus import crewai_event_bus
from crewai.events.event_listener import EventListener
from crewai.events.types.task_events import (
    TaskCompletedEvent,
    TaskFailedEvent,
    TaskStartedEvent,
)
from crewai.tasks.task_output import TaskOutput


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, role: str = "test_role"):
        self.role = role
        self.crew = MagicMock()


class MockTask:
    """Mock task for testing."""

    def __init__(self, task_id: str = "test_task"):
        self.id = task_id
        self.name = "Test Task"
        self.description = "A test task description"
        self.agent = MockAgent()


@pytest.fixture
def event_listener():
    """Create a fresh EventListener instance for testing."""
    EventListener._instance = None
    EventListener._initialized = False
    listener = EventListener()
    listener.setup_listeners(crewai_event_bus)
    return listener


@pytest.fixture
def mock_task():
    """Create a mock task for testing."""
    return MockTask()


@pytest.fixture
def mock_task_output():
    """Create a mock task output for testing."""
    return TaskOutput(
        description="Test task description",
        raw="Test output",
        agent="test_agent",
    )


@pytest.mark.asyncio
async def test_execution_spans_removed_on_task_completed(
    event_listener, mock_task, mock_task_output
):
    """Test that execution_spans entries are properly removed when a task completes.

    This test verifies the fix for the memory leak where completed tasks were
    setting execution_spans[source] = None instead of removing the key entirely.
    """
    with patch.object(event_listener._telemetry, "task_started") as mock_task_started:
        with patch.object(event_listener._telemetry, "task_ended"):
            mock_span = MagicMock()
            mock_task_started.return_value = mock_span

            start_event = TaskStartedEvent(context="test context", task=mock_task)
            future = crewai_event_bus.emit(mock_task, start_event)
            if future:
                await asyncio.wrap_future(future)

            assert mock_task in event_listener.execution_spans
            assert event_listener.execution_spans[mock_task] == mock_span

            completed_event = TaskCompletedEvent(output=mock_task_output, task=mock_task)
            future = crewai_event_bus.emit(mock_task, completed_event)
            if future:
                await asyncio.wrap_future(future)

            assert mock_task not in event_listener.execution_spans


@pytest.mark.asyncio
async def test_execution_spans_removed_on_task_failed(event_listener, mock_task):
    """Test that execution_spans entries are properly removed when a task fails.

    This test verifies the fix for the memory leak where failed tasks were
    setting execution_spans[source] = None instead of removing the key entirely.
    """
    with patch.object(event_listener._telemetry, "task_started") as mock_task_started:
        with patch.object(event_listener._telemetry, "task_ended"):
            mock_span = MagicMock()
            mock_task_started.return_value = mock_span

            start_event = TaskStartedEvent(context="test context", task=mock_task)
            future = crewai_event_bus.emit(mock_task, start_event)
            if future:
                await asyncio.wrap_future(future)

            assert mock_task in event_listener.execution_spans
            assert event_listener.execution_spans[mock_task] == mock_span

            failed_event = TaskFailedEvent(error="Test error", task=mock_task)
            future = crewai_event_bus.emit(mock_task, failed_event)
            if future:
                await asyncio.wrap_future(future)

            assert mock_task not in event_listener.execution_spans


@pytest.mark.asyncio
async def test_execution_spans_dict_size_does_not_grow_unbounded(
    event_listener, mock_task_output
):
    """Test that execution_spans dictionary size remains bounded after many tasks.

    This test simulates the memory leak scenario where many tasks complete/fail
    and verifies that the dictionary doesn't grow unboundedly.
    """
    num_tasks = 100

    with patch.object(event_listener._telemetry, "task_started") as mock_task_started:
        with patch.object(event_listener._telemetry, "task_ended"):
            mock_task_started.return_value = MagicMock()

            for i in range(num_tasks):
                task = MockTask(task_id=f"task_{i}")

                start_event = TaskStartedEvent(context="test context", task=task)
                future = crewai_event_bus.emit(task, start_event)
                if future:
                    await asyncio.wrap_future(future)

                if i % 2 == 0:
                    completed_event = TaskCompletedEvent(
                        output=mock_task_output, task=task
                    )
                    future = crewai_event_bus.emit(task, completed_event)
                else:
                    failed_event = TaskFailedEvent(error="Test error", task=task)
                    future = crewai_event_bus.emit(task, failed_event)

                if future:
                    await asyncio.wrap_future(future)

            assert len(event_listener.execution_spans) == 0


@pytest.mark.asyncio
async def test_execution_spans_handles_missing_task_gracefully(
    event_listener, mock_task, mock_task_output
):
    """Test that completing/failing a task not in execution_spans doesn't cause errors.

    This ensures the fix using pop(source, None) handles edge cases gracefully.
    """
    with patch.object(event_listener._telemetry, "task_ended"):
        assert mock_task not in event_listener.execution_spans

        completed_event = TaskCompletedEvent(output=mock_task_output, task=mock_task)
        future = crewai_event_bus.emit(mock_task, completed_event)
        if future:
            await asyncio.wrap_future(future)

        assert mock_task not in event_listener.execution_spans


@pytest.mark.asyncio
async def test_execution_spans_handles_missing_task_on_failure_gracefully(
    event_listener, mock_task
):
    """Test that failing a task not in execution_spans doesn't cause errors.

    This ensures the fix using pop(source, None) handles edge cases gracefully.
    """
    with patch.object(event_listener._telemetry, "task_ended"):
        assert mock_task not in event_listener.execution_spans

        failed_event = TaskFailedEvent(error="Test error", task=mock_task)
        future = crewai_event_bus.emit(mock_task, failed_event)
        if future:
            await asyncio.wrap_future(future)

        assert mock_task not in event_listener.execution_spans


@pytest.mark.asyncio
async def test_telemetry_task_ended_called_with_span_on_completion(
    event_listener, mock_task, mock_task_output
):
    """Test that telemetry.task_ended is called with the correct span on completion."""
    with patch.object(event_listener._telemetry, "task_started") as mock_task_started:
        with patch.object(event_listener._telemetry, "task_ended") as mock_task_ended:
            mock_span = MagicMock()
            mock_task_started.return_value = mock_span

            start_event = TaskStartedEvent(context="test context", task=mock_task)
            future = crewai_event_bus.emit(mock_task, start_event)
            if future:
                await asyncio.wrap_future(future)

            completed_event = TaskCompletedEvent(output=mock_task_output, task=mock_task)
            future = crewai_event_bus.emit(mock_task, completed_event)
            if future:
                await asyncio.wrap_future(future)

            mock_task_ended.assert_called_once_with(
                mock_span, mock_task, mock_task.agent.crew
            )


@pytest.mark.asyncio
async def test_telemetry_task_ended_called_with_span_on_failure(
    event_listener, mock_task
):
    """Test that telemetry.task_ended is called with the correct span on failure."""
    with patch.object(event_listener._telemetry, "task_started") as mock_task_started:
        with patch.object(event_listener._telemetry, "task_ended") as mock_task_ended:
            mock_span = MagicMock()
            mock_task_started.return_value = mock_span

            start_event = TaskStartedEvent(context="test context", task=mock_task)
            future = crewai_event_bus.emit(mock_task, start_event)
            if future:
                await asyncio.wrap_future(future)

            failed_event = TaskFailedEvent(error="Test error", task=mock_task)
            future = crewai_event_bus.emit(mock_task, failed_event)
            if future:
                await asyncio.wrap_future(future)

            mock_task_ended.assert_called_once_with(
                mock_span, mock_task, mock_task.agent.crew
            )
