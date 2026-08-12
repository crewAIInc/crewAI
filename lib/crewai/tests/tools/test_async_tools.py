"""Tests for async tool functionality."""

import asyncio

import pytest

from crewai.tools import BaseTool, tool
from crewai.tools.structured_tool import CrewStructuredTool


class SyncTool(BaseTool):
    """Test tool with synchronous _run method."""

    name: str = "sync_tool"
    description: str = "A synchronous tool for testing"

    def _run(self, input_text: str) -> str:
        """Process input text synchronously."""
        return f"Sync processed: {input_text}"


class AsyncTool(BaseTool):
    """Test tool with both sync and async implementations."""

    name: str = "async_tool"
    description: str = "An asynchronous tool for testing"

    def _run(self, input_text: str) -> str:
        """Process input text synchronously."""
        return f"Sync processed: {input_text}"

    async def _arun(self, input_text: str) -> str:
        """Process input text asynchronously."""
        await asyncio.sleep(0.01)
        return f"Async processed: {input_text}"


class TestBaseTool:
    """Tests for BaseTool async functionality."""

    def test_sync_tool_run_returns_result(self) -> None:
        """Test that sync tool run() returns correct result."""
        tool = SyncTool()
        result = tool.run(input_text="hello")
        assert result == "Sync processed: hello"

    def test_async_tool_run_returns_result(self) -> None:
        """Test that async tool run() works."""
        tool = AsyncTool()
        result = tool.run(input_text="hello")
        assert result == "Sync processed: hello"

    @pytest.mark.asyncio
    async def test_sync_tool_arun_raises_not_implemented(self) -> None:
        """Test that sync tool arun() raises NotImplementedError."""
        tool = SyncTool()
        with pytest.raises(NotImplementedError):
            await tool.arun(input_text="hello")

    @pytest.mark.asyncio
    async def test_async_tool_arun_returns_result(self) -> None:
        """Test that async tool arun() awaits directly."""
        tool = AsyncTool()
        result = await tool.arun(input_text="hello")
        assert result == "Async processed: hello"

    @pytest.mark.asyncio
    async def test_arun_increments_usage_count(self) -> None:
        """Test that arun increments the usage count."""
        tool = AsyncTool()
        assert tool.current_usage_count == 0

        await tool.arun(input_text="test")
        assert tool.current_usage_count == 1

        await tool.arun(input_text="test2")
        assert tool.current_usage_count == 2

    @pytest.mark.asyncio
    async def test_multiple_async_tools_run_concurrently(self) -> None:
        """Test that multiple async tools can run concurrently."""
        tool1 = AsyncTool()
        tool2 = AsyncTool()

        results = await asyncio.gather(
            tool1.arun(input_text="first"),
            tool2.arun(input_text="second"),
        )

        assert results[0] == "Async processed: first"
        assert results[1] == "Async processed: second"


class TestToolDecorator:
    """Tests for @tool decorator with async functions."""

    def test_sync_decorated_tool_run(self) -> None:
        """Test sync decorated tool works with run()."""

        @tool("sync_decorated")
        def sync_func(value: str) -> str:
            """A sync decorated tool."""
            return f"sync: {value}"

        result = sync_func.run(value="test")
        assert result == "sync: test"

    def test_async_decorated_tool_run(self) -> None:
        """Test async decorated tool works with run()."""

        @tool("async_decorated")
        async def async_func(value: str) -> str:
            """An async decorated tool."""
            await asyncio.sleep(0.01)
            return f"async: {value}"

        result = async_func.run(value="test")
        assert result == "async: test"

    @pytest.mark.asyncio
    async def test_sync_decorated_tool_arun_raises(self) -> None:
        """Test sync decorated tool arun() raises NotImplementedError."""

        @tool("sync_decorated_arun")
        def sync_func(value: str) -> str:
            """A sync decorated tool."""
            return f"sync: {value}"

        with pytest.raises(NotImplementedError):
            await sync_func.arun(value="test")

    @pytest.mark.asyncio
    async def test_async_decorated_tool_arun(self) -> None:
        """Test async decorated tool works with arun()."""

        @tool("async_decorated_arun")
        async def async_func(value: str) -> str:
            """An async decorated tool."""
            await asyncio.sleep(0.01)
            return f"async: {value}"

        result = await async_func.arun(value="test")
        assert result == "async: test"


class TestToolRunUnderRunningLoop:
    """Regression tests: running a tool whose ``_run`` returns a coroutine from
    inside a running event loop must return the result instead of raising
    ``RuntimeError: asyncio.run() cannot be called from a running event loop``.
    """

    def test_base_tool_run_inside_running_loop(self) -> None:
        """BaseTool.run() must resolve a coroutine result under a running loop."""

        class CoroReturningTool(BaseTool):
            name: str = "coro_returning"
            description: str = "Returns a coroutine from _run"

            def _run(self, value: str) -> str:
                async def _inner() -> str:
                    await asyncio.sleep(0.01)
                    return f"result: {value}"

                return _inner()  # type: ignore[return-value]

        async def caller() -> str:
            return CoroReturningTool().run(value="x")

        result = asyncio.run(caller())
        assert result == "result: x"

    def test_async_decorated_tool_run_inside_running_loop(self) -> None:
        """@tool with an async function must run under a running loop."""

        @tool("async_in_loop")
        async def async_func(value: str) -> str:
            """An async decorated tool."""
            await asyncio.sleep(0.01)
            return f"async: {value}"

        async def caller() -> str:
            return async_func.run(value="test")

        result = asyncio.run(caller())
        assert result == "async: test"

    def test_structured_tool_invoke_inside_running_loop(self) -> None:
        """CrewStructuredTool.invoke() must run an async func under a running loop."""

        async def async_func(value: str) -> str:
            """A structured async tool."""
            await asyncio.sleep(0.01)
            return f"structured: {value}"

        structured = CrewStructuredTool.from_function(
            func=async_func, name="structured_async"
        )

        async def caller() -> str:
            return structured.invoke({"value": "test"})

        result = asyncio.run(caller())
        assert result == "structured: test"


class TestAsyncToolWithIO:
    """Tests for async tools with simulated I/O operations."""

    @pytest.mark.asyncio
    async def test_async_tool_simulated_io(self) -> None:
        """Test async tool with simulated I/O delay."""

        class SlowAsyncTool(BaseTool):
            name: str = "slow_async"
            description: str = "Simulates slow I/O"

            def _run(self, delay: float) -> str:
                return f"Completed after {delay}s"

            async def _arun(self, delay: float) -> str:
                await asyncio.sleep(delay)
                return f"Completed after {delay}s"

        tool = SlowAsyncTool()
        result = await tool.arun(delay=0.05)
        assert result == "Completed after 0.05s"

    @pytest.mark.asyncio
    async def test_multiple_slow_tools_concurrent(self) -> None:
        """Test that slow async tools benefit from concurrency."""

        class SlowAsyncTool(BaseTool):
            name: str = "slow_async"
            description: str = "Simulates slow I/O"

            def _run(self, task_id: int, delay: float) -> str:
                return f"Task {task_id} done"

            async def _arun(self, task_id: int, delay: float) -> str:
                await asyncio.sleep(delay)
                return f"Task {task_id} done"

        tool = SlowAsyncTool()

        import time

        start = time.time()
        results = await asyncio.gather(
            tool.arun(task_id=1, delay=0.1),
            tool.arun(task_id=2, delay=0.1),
            tool.arun(task_id=3, delay=0.1),
        )
        elapsed = time.time() - start

        assert len(results) == 3
        assert all("done" in r for r in results)
        assert elapsed < 0.25, f"Expected concurrent execution, took {elapsed}s"