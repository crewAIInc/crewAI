"""Tests for CrewAgentExecutor state reset between task executions.

Verifies that invoke() and ainvoke() reset messages and iterations
at the start of each execution, preventing state leakage across
sequential tasks (issue #4389).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crewai.agents.crew_agent_executor import CrewAgentExecutor
from crewai.agents.parser import AgentFinish


@pytest.fixture
def mock_llm() -> MagicMock:
    """Create a mock LLM for testing."""
    llm = MagicMock()
    llm.supports_stop_words.return_value = True
    llm.stop = []
    return llm


@pytest.fixture
def mock_agent() -> MagicMock:
    """Create a mock agent for testing."""
    agent = MagicMock()
    agent.role = "Test Agent"
    agent.key = "test_agent_key"
    agent.verbose = False
    agent.id = "test_agent_id"
    return agent


@pytest.fixture
def mock_task() -> MagicMock:
    """Create a mock task for testing."""
    task = MagicMock()
    task.description = "Test task description"
    return task


@pytest.fixture
def mock_crew() -> MagicMock:
    """Create a mock crew for testing."""
    crew = MagicMock()
    crew.verbose = False
    crew._train = False
    return crew


@pytest.fixture
def mock_tools_handler() -> MagicMock:
    """Create a mock tools handler."""
    return MagicMock()


@pytest.fixture
def executor(
    mock_llm: MagicMock,
    mock_agent: MagicMock,
    mock_task: MagicMock,
    mock_crew: MagicMock,
    mock_tools_handler: MagicMock,
) -> CrewAgentExecutor:
    """Create a CrewAgentExecutor instance for testing."""
    return CrewAgentExecutor(
        llm=mock_llm,
        task=mock_task,
        crew=mock_crew,
        agent=mock_agent,
        prompt={"prompt": "Test prompt {input} {tool_names} {tools}"},
        max_iter=5,
        tools=[],
        tools_names="",
        stop_words=["Observation:"],
        tools_description="",
        tools_handler=mock_tools_handler,
    )


class TestInvokeResetsState:
    """Verify invoke() clears execution state before each run."""

    def test_messages_reset_on_invoke(self, executor: CrewAgentExecutor) -> None:
        """Messages from a previous execution must not leak into the next one."""
        # Simulate leftover state from a prior task execution
        executor.messages = [
            {"role": "system", "content": "old system prompt"},
            {"role": "user", "content": "old user prompt"},
            {"role": "assistant", "content": "old response"},
        ]
        executor.iterations = 7

        finish = AgentFinish(thought="done", output="result", text="result")

        with patch.object(executor, "_invoke_loop", return_value=finish),              patch.object(executor, "_create_short_term_memory"),              patch.object(executor, "_create_long_term_memory"),              patch.object(executor, "_create_external_memory"),              patch.object(executor, "_show_start_logs"):

            executor.invoke({"input": "new task", "tool_names": "", "tools": ""})

        # After invoke, messages should only contain the new task's prompts
        # (set up by _setup_messages), not the old ones.
        assert not any(
            msg.get("content") == "old system prompt" for msg in executor.messages
        ), "Old messages leaked into the new execution"

    def test_iterations_reset_on_invoke(self, executor: CrewAgentExecutor) -> None:
        """Iteration counter must start from zero for each new task."""
        executor.iterations = 10

        finish = AgentFinish(thought="done", output="result", text="result")

        with patch.object(executor, "_invoke_loop", return_value=finish),              patch.object(executor, "_create_short_term_memory"),              patch.object(executor, "_create_long_term_memory"),              patch.object(executor, "_create_external_memory"),              patch.object(executor, "_show_start_logs"):

            executor.invoke({"input": "new task", "tool_names": "", "tools": ""})

        # _invoke_loop is mocked so iterations stays at whatever invoke() set it to.
        # Since _invoke_loop is mocked (does nothing to iterations), final value
        # should be 0.
        assert executor.iterations == 0

    def test_sequential_invokes_are_isolated(
        self, executor: CrewAgentExecutor
    ) -> None:
        """Two sequential invoke() calls must not share any execution state."""
        call_count = 0
        captured_messages: list[list] = []
        captured_iterations: list[int] = []

        def tracking_invoke_loop() -> AgentFinish:
            nonlocal call_count
            call_count += 1
            # Capture the state at the start of the loop
            captured_messages.append(list(executor.messages))
            captured_iterations.append(executor.iterations)
            # Simulate the loop adding messages and incrementing iterations
            executor.messages.append(
                {"role": "assistant", "content": f"response_{call_count}"}
            )
            executor.iterations = 3
            return AgentFinish(
                thought="done",
                output=f"result_{call_count}",
                text=f"result_{call_count}",
            )

        with patch.object(
            executor, "_invoke_loop", side_effect=tracking_invoke_loop
        ),              patch.object(executor, "_create_short_term_memory"),              patch.object(executor, "_create_long_term_memory"),              patch.object(executor, "_create_external_memory"),              patch.object(executor, "_show_start_logs"):

            executor.invoke({"input": "task 1", "tool_names": "", "tools": ""})
            executor.invoke({"input": "task 2", "tool_names": "", "tools": ""})

        assert call_count == 2

        # Second invoke should have started with fresh messages (only the new prompt)
        # not carrying over the "response_1" from the first invoke
        assert not any(
            msg.get("content") == "response_1" for msg in captured_messages[1]
        ), "Messages from first invoke leaked into second invoke"

        # Second invoke should have started with iterations == 0
        assert captured_iterations[1] == 0, (
            f"Iterations were {captured_iterations[1]} at start of second invoke, expected 0"
        )


class TestAinvokeResetsState:
    """Verify ainvoke() clears execution state before each run."""

    @pytest.mark.asyncio
    async def test_messages_reset_on_ainvoke(
        self, executor: CrewAgentExecutor
    ) -> None:
        """Messages from a previous execution must not leak into async execution."""
        executor.messages = [
            {"role": "system", "content": "old system prompt"},
            {"role": "user", "content": "old user prompt"},
        ]
        executor.iterations = 5

        finish = AgentFinish(thought="done", output="result", text="result")

        with patch.object(
            executor, "_ainvoke_loop", new_callable=AsyncMock, return_value=finish
        ),              patch.object(
                 executor, "_ainject_multimodal_files", new_callable=AsyncMock
             ),              patch.object(executor, "_create_short_term_memory"),              patch.object(executor, "_create_long_term_memory"),              patch.object(executor, "_create_external_memory"),              patch.object(executor, "_show_start_logs"):

            await executor.ainvoke(
                {"input": "new task", "tool_names": "", "tools": ""}
            )

        assert not any(
            msg.get("content") == "old system prompt" for msg in executor.messages
        ), "Old messages leaked into async execution"

    @pytest.mark.asyncio
    async def test_iterations_reset_on_ainvoke(
        self, executor: CrewAgentExecutor
    ) -> None:
        """Iteration counter must start from zero for async execution."""
        executor.iterations = 10

        finish = AgentFinish(thought="done", output="result", text="result")

        with patch.object(
            executor, "_ainvoke_loop", new_callable=AsyncMock, return_value=finish
        ),              patch.object(
                 executor, "_ainject_multimodal_files", new_callable=AsyncMock
             ),              patch.object(executor, "_create_short_term_memory"),              patch.object(executor, "_create_long_term_memory"),              patch.object(executor, "_create_external_memory"),              patch.object(executor, "_show_start_logs"):

            await executor.ainvoke(
                {"input": "new task", "tool_names": "", "tools": ""}
            )

        assert executor.iterations == 0
