"""Tests for Python 3.14 compatibility.

Python 3.14 changed asyncio.get_event_loop() to raise RuntimeError when no
running event loop exists instead of creating one. All async code paths must
use asyncio.get_running_loop() instead.

See: https://github.com/crewAIInc/crewAI/issues/5109
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crewai.tools.structured_tool import CrewStructuredTool
from crewai.utilities.streaming import create_streaming_state


class TestStructuredToolAsyncCompat:
    """Test that CrewStructuredTool.ainvoke uses get_running_loop correctly."""

    @pytest.mark.asyncio
    async def test_ainvoke_sync_func_uses_running_loop(self) -> None:
        """ainvoke() with a sync function must use the running event loop."""

        def sync_func(x: int) -> int:
            """A sync function."""
            return x * 2

        tool = CrewStructuredTool.from_function(
            func=sync_func, name="double", description="Doubles a number"
        )
        result = await tool.ainvoke({"x": 5})
        assert result == 10

    @pytest.mark.asyncio
    async def test_ainvoke_async_func(self) -> None:
        """ainvoke() with an async function should call it directly."""

        async def async_func(x: int) -> int:
            """An async function."""
            return x * 3

        tool = CrewStructuredTool.from_function(
            func=async_func, name="triple", description="Triples a number"
        )
        result = await tool.ainvoke({"x": 4})
        assert result == 12

    @pytest.mark.asyncio
    async def test_ainvoke_sync_func_runs_in_executor(self) -> None:
        """Verify ainvoke offloads sync functions to an executor via the running loop."""
        import threading

        call_thread_ids: list[int] = []

        def sync_func(x: int) -> int:
            """A sync function that records its thread."""
            call_thread_ids.append(threading.current_thread().ident or 0)
            return x + 1

        tool = CrewStructuredTool.from_function(
            func=sync_func, name="inc", description="Increment"
        )

        result = await tool.ainvoke({"x": 1})
        assert result == 2
        assert len(call_thread_ids) == 1
        # Sync func should run in a different thread (executor)
        assert call_thread_ids[0] != threading.current_thread().ident


class TestStreamingStateAsyncCompat:
    """Test that create_streaming_state uses get_running_loop correctly."""

    @pytest.mark.asyncio
    async def test_create_streaming_state_async_uses_running_loop(self) -> None:
        """create_streaming_state(use_async=True) must use the running loop."""
        task_info = {
            "index": 0,
            "name": "test",
            "id": "test-id",
            "agent_role": "tester",
            "agent_id": "agent-id",
        }
        state = create_streaming_state(
            current_task_info=task_info,
            result_holder=[],
            use_async=True,
        )
        assert state.loop is not None
        assert state.async_queue is not None
        assert state.loop is asyncio.get_running_loop()

    def test_create_streaming_state_sync_no_loop_needed(self) -> None:
        """create_streaming_state(use_async=False) should not require a loop."""
        task_info = {
            "index": 0,
            "name": "test",
            "id": "test-id",
            "agent_role": "tester",
            "agent_id": "agent-id",
        }
        state = create_streaming_state(
            current_task_info=task_info,
            result_holder=[],
            use_async=False,
        )
        assert state.loop is None
        assert state.async_queue is None

    @pytest.mark.asyncio
    async def test_create_streaming_state_async_uses_get_running_loop_not_get_event_loop(
        self,
    ) -> None:
        """Verify create_streaming_state does not call asyncio.get_event_loop()."""
        task_info = {
            "index": 0,
            "name": "test",
            "id": "test-id",
            "agent_role": "tester",
            "agent_id": "agent-id",
        }

        with patch("crewai.utilities.streaming.asyncio") as mock_asyncio:
            mock_asyncio.Queue = asyncio.Queue
            mock_asyncio.get_running_loop.return_value = asyncio.get_running_loop()

            create_streaming_state(
                current_task_info=task_info,
                result_holder=[],
                use_async=True,
            )

            mock_asyncio.get_running_loop.assert_called_once()
            mock_asyncio.get_event_loop.assert_not_called()


class TestChromaDBClientAsyncCompat:
    """Test that ChromaDBClient._alocked uses get_running_loop correctly."""

    @pytest.mark.asyncio
    async def test_alocked_without_lock_name(self) -> None:
        """_alocked should yield immediately when no lock name is set."""
        from crewai.rag.chromadb.client import ChromaDBClient

        mock_client = MagicMock()
        mock_ef = MagicMock()
        client = ChromaDBClient(
            client=mock_client,
            embedding_function=mock_ef,
            lock_name=None,
        )

        async with client._alocked():
            pass  # Should not raise

    @pytest.mark.asyncio
    async def test_alocked_uses_get_running_loop_not_get_event_loop(self) -> None:
        """Verify _alocked does not call asyncio.get_event_loop()."""
        from crewai.rag.chromadb.client import ChromaDBClient

        mock_client = MagicMock()
        mock_ef = MagicMock()
        client = ChromaDBClient(
            client=mock_client,
            embedding_function=mock_ef,
            lock_name="test-lock",
        )

        with patch("crewai.rag.chromadb.client.asyncio") as mock_asyncio:
            loop = asyncio.get_running_loop()
            mock_asyncio.get_running_loop.return_value = loop

            mock_cm = MagicMock()
            with patch("crewai.rag.chromadb.client.store_lock", return_value=mock_cm):
                async with client._alocked():
                    pass

            mock_asyncio.get_running_loop.assert_called()
            mock_asyncio.get_event_loop.assert_not_called()


class TestGetRunningLoopInAsyncContext:
    """General tests ensuring get_running_loop works in async contexts."""

    @pytest.mark.asyncio
    async def test_get_running_loop_available_in_async_context(self) -> None:
        """asyncio.get_running_loop() should work in an async context."""
        loop = asyncio.get_running_loop()
        assert loop is not None
        assert loop.is_running()

    @pytest.mark.asyncio
    async def test_run_in_executor_with_running_loop(self) -> None:
        """run_in_executor should work with get_running_loop()."""
        loop = asyncio.get_running_loop()

        def sync_work() -> str:
            return "done"

        result = await loop.run_in_executor(None, sync_work)
        assert result == "done"

    def test_get_running_loop_raises_outside_async(self) -> None:
        """get_running_loop() should raise RuntimeError outside async context."""
        with pytest.raises(RuntimeError):
            asyncio.get_running_loop()
