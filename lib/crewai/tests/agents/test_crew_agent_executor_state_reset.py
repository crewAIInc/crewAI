"""Tests for CrewAgentExecutor state reset between task executions.

Verifies that messages and iterations are properly reset when the executor
is reused across multiple tasks, preventing state leakage between executions.
"""

from unittest.mock import Mock, patch, MagicMock

import pytest

from crewai.agents.crew_agent_executor import CrewAgentExecutor
from crewai.agents.parser import AgentFinish


class TestCrewAgentExecutorStateReset:
    """Test that CrewAgentExecutor properly resets state between invocations."""

    @pytest.fixture
    def mock_executor(self):
        """Create a CrewAgentExecutor with mocked dependencies."""
        llm = Mock()
        llm.supports_stop_words.return_value = True
        llm.stop = []

        task = Mock()
        task.description = "Test task"
        task.human_input = False
        task.id = "test-task-id"
        task.increment_tools_errors = Mock()

        crew = Mock()
        crew.verbose = False
        crew._train = False
        crew.id = "test-crew-id"

        agent = Mock()
        agent.id = "test-agent-id"
        agent.role = "Test Agent"
        agent.verbose = False
        agent.key = "test-key"

        prompt = {"prompt": "Test prompt with {input}"}

        tools_handler = Mock()
        tools_handler.cache = None

        executor = CrewAgentExecutor(
            llm=llm,
            task=task,
            crew=crew,
            agent=agent,
            prompt=prompt,
            max_iter=10,
            tools=[],
            tools_names="",
            stop_words=["Observation"],
            tools_description="",
            tools_handler=tools_handler,
        )

        return executor

    def test_messages_reset_on_invoke(self, mock_executor):
        """Test that messages are reset at the beginning of invoke()."""
        # Simulate leftover messages from a previous execution
        mock_executor.messages = [
            {"role": "user", "content": "Previous task message"},
            {"role": "assistant", "content": "Previous task response"},
        ]
        mock_executor.iterations = 5

        # Mock the execution loop to return immediately
        with patch.object(
            mock_executor, '_invoke_loop',
            return_value=AgentFinish(thought="done", output="result", text="result")
        ):
            with patch.object(mock_executor, '_create_short_term_memory'):
                with patch.object(mock_executor, '_create_long_term_memory'):
                    with patch.object(mock_executor, '_create_external_memory'):
                        mock_executor.invoke({"input": "New task"})

        # After invoke, messages should have been reset (not contain old messages)
        # The new messages should only contain the new task prompt
        assert len(mock_executor.messages) >= 1
        assert all(
            "Previous task" not in str(msg.get("content", ""))
            for msg in mock_executor.messages
        ), "Messages from previous execution should be cleared"

    def test_iterations_reset_on_invoke(self, mock_executor):
        """Test that iterations counter is reset at the beginning of invoke()."""
        # Simulate leftover iterations from a previous execution
        mock_executor.iterations = 7

        # Mock the execution loop to return immediately
        with patch.object(
            mock_executor, '_invoke_loop',
            return_value=AgentFinish(thought="done", output="result", text="result")
        ):
            with patch.object(mock_executor, '_create_short_term_memory'):
                with patch.object(mock_executor, '_create_long_term_memory'):
                    with patch.object(mock_executor, '_create_external_memory'):
                        # Capture iterations value after reset but before loop
                        original_invoke_loop = mock_executor._invoke_loop
                        iterations_after_reset = []
                        
                        def capture_iterations(*args, **kwargs):
                            iterations_after_reset.append(mock_executor.iterations)
                            return AgentFinish(thought="done", output="result", text="result")
                        
                        with patch.object(mock_executor, '_invoke_loop', side_effect=capture_iterations):
                            mock_executor.invoke({"input": "New task"})

        # iterations should have been reset to 0 before the loop started
        assert iterations_after_reset[0] == 0, "Iterations should be reset to 0 at start of invoke"


@pytest.mark.asyncio
class TestCrewAgentExecutorAsyncStateReset:
    """Test that CrewAgentExecutor properly resets state in async invocations."""

    @pytest.fixture
    def mock_executor(self):
        """Create a CrewAgentExecutor with mocked dependencies."""
        llm = Mock()
        llm.supports_stop_words.return_value = True
        llm.stop = []

        task = Mock()
        task.description = "Test task"
        task.human_input = False
        task.id = "test-task-id"
        task.increment_tools_errors = Mock()

        crew = Mock()
        crew.verbose = False
        crew._train = False
        crew.id = "test-crew-id"

        agent = Mock()
        agent.id = "test-agent-id"
        agent.role = "Test Agent"
        agent.verbose = False
        agent.key = "test-key"

        prompt = {"prompt": "Test prompt with {input}"}

        tools_handler = Mock()
        tools_handler.cache = None

        executor = CrewAgentExecutor(
            llm=llm,
            task=task,
            crew=crew,
            agent=agent,
            prompt=prompt,
            max_iter=10,
            tools=[],
            tools_names="",
            stop_words=["Observation"],
            tools_description="",
            tools_handler=tools_handler,
        )

        return executor

    async def test_messages_reset_on_ainvoke(self, mock_executor):
        """Test that messages are reset at the beginning of ainvoke()."""
        # Simulate leftover messages from a previous execution
        mock_executor.messages = [
            {"role": "user", "content": "Previous async task message"},
            {"role": "assistant", "content": "Previous async task response"},
        ]
        mock_executor.iterations = 5

        # Mock the async execution loop to return immediately
        async def mock_ainvoke_loop():
            return AgentFinish(thought="done", output="result", text="result")

        with patch.object(mock_executor, '_ainvoke_loop', side_effect=mock_ainvoke_loop):
            with patch.object(mock_executor, '_create_short_term_memory'):
                with patch.object(mock_executor, '_create_long_term_memory'):
                    with patch.object(mock_executor, '_create_external_memory'):
                        await mock_executor.ainvoke({"input": "New async task"})

        # After ainvoke, messages should have been reset (not contain old messages)
        assert len(mock_executor.messages) >= 1
        assert all(
            "Previous async task" not in str(msg.get("content", ""))
            for msg in mock_executor.messages
        ), "Messages from previous execution should be cleared"

    async def test_iterations_reset_on_ainvoke(self, mock_executor):
        """Test that iterations counter is reset at the beginning of ainvoke()."""
        # Simulate leftover iterations from a previous execution
        mock_executor.iterations = 7

        iterations_after_reset = []

        async def capture_iterations():
            iterations_after_reset.append(mock_executor.iterations)
            return AgentFinish(thought="done", output="result", text="result")

        with patch.object(mock_executor, '_ainvoke_loop', side_effect=capture_iterations):
            with patch.object(mock_executor, '_create_short_term_memory'):
                with patch.object(mock_executor, '_create_long_term_memory'):
                    with patch.object(mock_executor, '_create_external_memory'):
                        await mock_executor.ainvoke({"input": "New async task"})

        # iterations should have been reset to 0 before the loop started
        assert iterations_after_reset[0] == 0, "Iterations should be reset to 0 at start of ainvoke"
