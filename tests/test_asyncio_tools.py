"""Tests for asyncio tool execution in different contexts."""

import asyncio
from unittest.mock import patch

import pytest

from crewai import Agent, Crew, Task
from crewai.tools import tool


@tool
async def async_test_tool(message: str) -> str:
    """An async tool that returns a message."""
    await asyncio.sleep(0.01)
    return f"Processed: {message}"


@tool
def sync_test_tool(message: str) -> str:
    """A sync tool that returns a message."""
    return f"Sync: {message}"


class TestAsyncioToolExecution:
    """Test that tools work correctly in different asyncio contexts."""

    @patch("crewai.Agent.execute_task")
    def test_async_tool_with_asyncio_to_thread(self, mock_execute_task):
        """Test that async tools work when crew is run with asyncio.to_thread."""
        mock_execute_task.return_value = "Task completed"

        agent = Agent(
            role="Test Agent",
            goal="Test async tool execution",
            backstory="A test agent",
        )

        task = Task(
            description="Test task",
            expected_output="A result",
            agent=agent,
            tools=[async_test_tool],
        )

        crew = Crew(agents=[agent], tasks=[task], verbose=False)

        async def run_with_to_thread():
            """Run crew with asyncio.to_thread - this should not hang."""
            result = await asyncio.to_thread(crew.kickoff)
            return result

        result = asyncio.run(run_with_to_thread())
        assert result is not None

    @patch("crewai.Agent.execute_task")
    def test_sync_tool_with_asyncio_to_thread(self, mock_execute_task):
        """Test that sync tools work when crew is run with asyncio.to_thread."""
        mock_execute_task.return_value = "Task completed"

        agent = Agent(
            role="Test Agent",
            goal="Test sync tool execution",
            backstory="A test agent",
        )

        task = Task(
            description="Test task",
            expected_output="A result",
            agent=agent,
            tools=[sync_test_tool],
        )

        crew = Crew(agents=[agent], tasks=[task], verbose=False)

        async def run_with_to_thread():
            """Run crew with asyncio.to_thread."""
            result = await asyncio.to_thread(crew.kickoff)
            return result

        result = asyncio.run(run_with_to_thread())
        assert result is not None

    @pytest.mark.asyncio
    @patch("crewai.Agent.execute_task")
    async def test_async_tool_with_kickoff_async(self, mock_execute_task):
        """Test that async tools work with kickoff_async."""
        mock_execute_task.return_value = "Task completed"

        agent = Agent(
            role="Test Agent",
            goal="Test async tool execution",
            backstory="A test agent",
        )

        task = Task(
            description="Test task",
            expected_output="A result",
            agent=agent,
            tools=[async_test_tool],
        )

        crew = Crew(agents=[agent], tasks=[task], verbose=False)

        result = await crew.kickoff_async()
        assert result is not None

    def test_async_tool_direct_invocation(self):
        """Test that async tools can be invoked directly from sync context."""
        structured_tool = async_test_tool.to_structured_tool()
        result = structured_tool.invoke({"message": "test"})
        assert result == "Processed: test"

    def test_async_tool_invocation_from_thread(self):
        """Test that async tools work when invoked from a thread pool."""
        structured_tool = async_test_tool.to_structured_tool()

        def invoke_tool():
            return structured_tool.invoke({"message": "test"})

        async def run_in_thread():
            result = await asyncio.to_thread(invoke_tool)
            return result

        result = asyncio.run(run_in_thread())
        assert result == "Processed: test"

    @pytest.mark.asyncio
    async def test_multiple_async_tools_concurrent(self):
        """Test multiple async tool invocations concurrently."""
        structured_tool = async_test_tool.to_structured_tool()

        async def invoke_async():
            return await structured_tool.ainvoke({"message": "test"})

        results = await asyncio.gather(
            invoke_async(), invoke_async(), invoke_async()
        )

        assert len(results) == 3
        for r in results:
            assert "test" in str(r)
