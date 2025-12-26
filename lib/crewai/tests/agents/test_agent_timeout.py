"""Test Agent timeout handling and cooperative cancellation."""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from crewai import Agent, Task
from crewai.agents.crew_agent_executor import CrewAgentExecutor


class TestExecutorDeadline:
    """Tests for CrewAgentExecutor deadline functionality."""

    def test_set_execution_deadline(self):
        """Test that set_execution_deadline sets the deadline correctly."""
        executor = MagicMock(spec=CrewAgentExecutor)
        executor._execution_deadline = None

        CrewAgentExecutor.set_execution_deadline(executor, 5)

        assert executor._execution_deadline is not None
        assert executor._execution_deadline > time.monotonic()

    def test_clear_execution_deadline(self):
        """Test that clear_execution_deadline clears the deadline."""
        executor = MagicMock(spec=CrewAgentExecutor)
        executor._execution_deadline = time.monotonic() + 100

        CrewAgentExecutor.clear_execution_deadline(executor)

        assert executor._execution_deadline is None

    def test_check_execution_deadline_not_exceeded(self):
        """Test that _check_execution_deadline does not raise when deadline not exceeded."""
        executor = MagicMock(spec=CrewAgentExecutor)
        executor._execution_deadline = time.monotonic() + 100
        executor.task = MagicMock()
        executor.task.description = "Test task"

        CrewAgentExecutor._check_execution_deadline(executor)

    def test_check_execution_deadline_exceeded(self):
        """Test that _check_execution_deadline raises TimeoutError when deadline exceeded."""
        executor = MagicMock(spec=CrewAgentExecutor)
        executor._execution_deadline = time.monotonic() - 1
        executor.task = MagicMock()
        executor.task.description = "Test task"

        with pytest.raises(TimeoutError) as exc_info:
            CrewAgentExecutor._check_execution_deadline(executor)

        assert "Test task" in str(exc_info.value)
        assert "timed out" in str(exc_info.value)

    def test_check_execution_deadline_no_deadline_set(self):
        """Test that _check_execution_deadline does nothing when no deadline is set."""
        executor = MagicMock(spec=CrewAgentExecutor)
        executor._execution_deadline = None
        executor.task = MagicMock()
        executor.task.description = "Test task"

        CrewAgentExecutor._check_execution_deadline(executor)


class TestAgentTimeoutBehavior:
    """Tests for Agent timeout behavior."""

    def test_execute_with_timeout_sets_deadline(self):
        """Test that _execute_with_timeout sets the deadline on the executor."""
        agent = Agent(
            role="test role",
            goal="test goal",
            backstory="test backstory",
            max_execution_time=5,
        )

        mock_executor = MagicMock()
        mock_executor.invoke.return_value = {"output": "test output"}
        agent.agent_executor = mock_executor

        task = Task(
            description="Test task",
            expected_output="Test output",
            agent=agent,
        )

        with patch.object(agent, "_execute_without_timeout", return_value="test output"):
            agent._execute_with_timeout("test prompt", task, 5)

        mock_executor.set_execution_deadline.assert_called_once_with(5)
        mock_executor.clear_execution_deadline.assert_called_once()

    def test_execute_with_timeout_clears_deadline_on_success(self):
        """Test that _execute_with_timeout clears the deadline after successful execution."""
        agent = Agent(
            role="test role",
            goal="test goal",
            backstory="test backstory",
            max_execution_time=5,
        )

        mock_executor = MagicMock()
        agent.agent_executor = mock_executor

        task = Task(
            description="Test task",
            expected_output="Test output",
            agent=agent,
        )

        with patch.object(agent, "_execute_without_timeout", return_value="test output"):
            result = agent._execute_with_timeout("test prompt", task, 5)

        assert result == "test output"
        mock_executor.clear_execution_deadline.assert_called_once()

    def test_execute_with_timeout_clears_deadline_on_timeout(self):
        """Test that _execute_with_timeout clears the deadline even when timeout occurs."""
        agent = Agent(
            role="test role",
            goal="test goal",
            backstory="test backstory",
            max_execution_time=1,
        )

        mock_executor = MagicMock()
        agent.agent_executor = mock_executor

        task = Task(
            description="Test task",
            expected_output="Test output",
            agent=agent,
        )

        def slow_execution(*args, **kwargs):
            time.sleep(5)
            return "test output"

        with patch.object(agent, "_execute_without_timeout", side_effect=slow_execution):
            with pytest.raises(TimeoutError):
                agent._execute_with_timeout("test prompt", task, 1)

        mock_executor.clear_execution_deadline.assert_called_once()

    def test_execute_with_timeout_raises_timeout_error(self):
        """Test that _execute_with_timeout raises TimeoutError when execution exceeds timeout."""
        agent = Agent(
            role="test role",
            goal="test goal",
            backstory="test backstory",
            max_execution_time=1,
        )

        mock_executor = MagicMock()
        agent.agent_executor = mock_executor

        task = Task(
            description="Test task",
            expected_output="Test output",
            agent=agent,
        )

        def slow_execution(*args, **kwargs):
            time.sleep(5)
            return "test output"

        with patch.object(agent, "_execute_without_timeout", side_effect=slow_execution):
            with pytest.raises(TimeoutError) as exc_info:
                agent._execute_with_timeout("test prompt", task, 1)

        assert "Test task" in str(exc_info.value)
        assert "timed out" in str(exc_info.value)


class TestCooperativeCancellation:
    """Tests for cooperative cancellation behavior."""

    def test_timeout_returns_control_promptly(self):
        """Test that timeout returns control to caller promptly (within reasonable bounds)."""
        agent = Agent(
            role="test role",
            goal="test goal",
            backstory="test backstory",
            max_execution_time=1,
        )

        mock_executor = MagicMock()
        agent.agent_executor = mock_executor

        task = Task(
            description="Test task",
            expected_output="Test output",
            agent=agent,
        )

        execution_started = threading.Event()

        def slow_execution(*args, **kwargs):
            execution_started.set()
            time.sleep(10)
            return "test output"

        with patch.object(agent, "_execute_without_timeout", side_effect=slow_execution):
            start_time = time.monotonic()
            with pytest.raises(TimeoutError):
                agent._execute_with_timeout("test prompt", task, 1)
            elapsed_time = time.monotonic() - start_time

        assert elapsed_time < 3, f"Timeout should return control within 3 seconds, took {elapsed_time:.2f}s"

    def test_executor_deadline_checked_in_invoke_loop(self):
        """Test that the executor checks the deadline in the invoke loop."""
        mock_llm = MagicMock()
        mock_llm.supports_stop_words.return_value = False
        mock_llm.call.return_value = "Final Answer: test"

        mock_task = MagicMock()
        mock_task.description = "Test task"

        mock_crew = MagicMock()
        mock_crew.verbose = False

        mock_agent = MagicMock()
        mock_agent.verbose = False
        mock_agent.role = "Test Agent"

        executor = CrewAgentExecutor(
            llm=mock_llm,
            task=mock_task,
            crew=mock_crew,
            agent=mock_agent,
            prompt={"prompt": "test"},
            max_iter=10,
            tools=[],
            tools_names="",
            stop_words=[],
            tools_description="",
            tools_handler=MagicMock(),
        )

        executor.set_execution_deadline(0.001)
        time.sleep(0.01)

        with pytest.raises(TimeoutError) as exc_info:
            executor.invoke({"input": "test", "tool_names": "", "tools": ""})

        assert "timed out" in str(exc_info.value)


class TestAsyncTimeoutBehavior:
    """Tests for async timeout behavior."""

    @pytest.mark.asyncio
    async def test_aexecute_with_timeout_sets_deadline(self):
        """Test that _aexecute_with_timeout sets the deadline on the executor."""
        agent = Agent(
            role="test role",
            goal="test goal",
            backstory="test backstory",
            max_execution_time=5,
        )

        mock_executor = MagicMock()
        mock_executor.ainvoke = MagicMock(return_value={"output": "test output"})
        agent.agent_executor = mock_executor

        task = Task(
            description="Test task",
            expected_output="Test output",
            agent=agent,
        )

        async def mock_aexecute(*args, **kwargs):
            return "test output"

        with patch.object(agent, "_aexecute_without_timeout", side_effect=mock_aexecute):
            await agent._aexecute_with_timeout("test prompt", task, 5)

        mock_executor.set_execution_deadline.assert_called_once_with(5)
        mock_executor.clear_execution_deadline.assert_called_once()

    @pytest.mark.asyncio
    async def test_aexecute_with_timeout_clears_deadline_on_timeout(self):
        """Test that _aexecute_with_timeout clears the deadline even when timeout occurs."""
        import asyncio

        agent = Agent(
            role="test role",
            goal="test goal",
            backstory="test backstory",
            max_execution_time=1,
        )

        mock_executor = MagicMock()
        agent.agent_executor = mock_executor

        task = Task(
            description="Test task",
            expected_output="Test output",
            agent=agent,
        )

        async def slow_execution(*args, **kwargs):
            await asyncio.sleep(5)
            return "test output"

        with patch.object(agent, "_aexecute_without_timeout", side_effect=slow_execution):
            with pytest.raises(TimeoutError):
                await agent._aexecute_with_timeout("test prompt", task, 1)

        mock_executor.clear_execution_deadline.assert_called_once()
