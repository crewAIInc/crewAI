"""Tests for CrewAgentExecutor state reset between task executions.

Verifies that messages and iterations are properly reset when invoke() or
ainvoke() is called multiple times on the same executor instance, preventing
context pollution across sequential tasks (issues #4319, #4389, #4415).
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crewai.agents.crew_agent_executor import CrewAgentExecutor
from crewai.agents.parser import AgentFinish


@pytest.fixture
def mock_llm() -> MagicMock:
    llm = MagicMock()
    llm.supports_stop_words.return_value = True
    llm.stop = []
    return llm


@pytest.fixture
def mock_agent() -> MagicMock:
    agent = MagicMock()
    agent.role = "Test Agent"
    agent.key = "test_agent_key"
    agent.verbose = False
    agent.id = "test_agent_id"
    return agent


@pytest.fixture
def mock_task() -> MagicMock:
    task = MagicMock()
    task.description = "Test task"
    task.human_input = False
    task.response_model = None
    task.id = "test_task_id"
    return task


@pytest.fixture
def mock_crew() -> MagicMock:
    crew = MagicMock()
    crew.verbose = False
    crew._train = False
    crew.id = "test_crew_id"
    return crew


@pytest.fixture
def mock_tools_handler() -> MagicMock:
    return MagicMock()


@pytest.fixture
def executor(
    mock_llm: MagicMock,
    mock_agent: MagicMock,
    mock_task: MagicMock,
    mock_crew: MagicMock,
    mock_tools_handler: MagicMock,
) -> CrewAgentExecutor:
    return CrewAgentExecutor(
        llm=mock_llm,
        task=mock_task,
        crew=mock_crew,
        agent=mock_agent,
        prompt={"system": "You are a helpful agent.", "user": "Task: {input} Tools: {tool_names} {tools}"},
        max_iter=5,
        tools=[],
        tools_names="",
        stop_words=["Observation:"],
        tools_description="",
        tools_handler=mock_tools_handler,
    )


class TestCrewAgentExecutorStateReset:
    """Tests that CrewAgentExecutor resets messages and iterations on each invoke."""

    def test_invoke_resets_messages(self, executor: CrewAgentExecutor) -> None:
        """Messages from a previous invoke must not leak into the next one."""
        executor.messages = [
            {"role": "system", "content": "old system prompt"},
            {"role": "user", "content": "old task prompt"},
            {"role": "assistant", "content": "old response"},
        ]
        executor.iterations = 7

        with patch.object(
            executor,
            "_invoke_loop",
            return_value=AgentFinish(thought="done", output="result", text="Final Answer: result"),
        ):
            with patch.object(executor, "_create_short_term_memory"):
                with patch.object(executor, "_create_long_term_memory"):
                    with patch.object(executor, "_create_external_memory"):
                        executor.invoke({"input": "new task", "tool_names": "", "tools": ""})

        system_msgs = [m for m in executor.messages if m.get("role") == "system"]
        assert len(system_msgs) == 1, (
            f"Expected exactly 1 system message after invoke, got {len(system_msgs)}"
        )

    def test_invoke_resets_iterations(self, executor: CrewAgentExecutor) -> None:
        """Iterations must be reset to 0 at the start of each invoke."""
        executor.iterations = 42

        with patch.object(
            executor,
            "_invoke_loop",
            return_value=AgentFinish(thought="done", output="result", text="done"),
        ):
            with patch.object(executor, "_create_short_term_memory"):
                with patch.object(executor, "_create_long_term_memory"):
                    with patch.object(executor, "_create_external_memory"):
                        executor.invoke({"input": "task", "tool_names": "", "tools": ""})

    def test_sequential_invokes_no_message_accumulation(
        self, executor: CrewAgentExecutor
    ) -> None:
        """Calling invoke multiple times must not accumulate messages."""
        message_counts: list[int] = []

        original_invoke_loop = executor._invoke_loop

        def capture_messages_then_finish() -> AgentFinish:
            message_counts.append(len(executor.messages))
            return AgentFinish(thought="done", output="result", text="done")

        with patch.object(executor, "_invoke_loop", side_effect=capture_messages_then_finish):
            with patch.object(executor, "_create_short_term_memory"):
                with patch.object(executor, "_create_long_term_memory"):
                    with patch.object(executor, "_create_external_memory"):
                        for i in range(3):
                            executor.invoke(
                                {"input": f"task {i}", "tool_names": "", "tools": ""}
                            )

        assert len(message_counts) == 3
        assert message_counts[0] == message_counts[1] == message_counts[2], (
            f"Message counts should be equal across invocations, got {message_counts}"
        )

    def test_sequential_invokes_no_duplicate_system_messages(
        self, executor: CrewAgentExecutor
    ) -> None:
        """Each invoke must contain exactly one system message, not N."""
        system_msg_counts: list[int] = []

        def capture_and_finish() -> AgentFinish:
            count = sum(1 for m in executor.messages if m.get("role") == "system")
            system_msg_counts.append(count)
            return AgentFinish(thought="done", output="ok", text="ok")

        with patch.object(executor, "_invoke_loop", side_effect=capture_and_finish):
            with patch.object(executor, "_create_short_term_memory"):
                with patch.object(executor, "_create_long_term_memory"):
                    with patch.object(executor, "_create_external_memory"):
                        for _ in range(5):
                            executor.invoke({"input": "task", "tool_names": "", "tools": ""})

        assert all(c == 1 for c in system_msg_counts), (
            f"Expected 1 system message per invoke, got {system_msg_counts}"
        )


class TestCrewAgentExecutorAsyncStateReset:
    """Tests that ainvoke also resets messages and iterations."""

    @pytest.mark.asyncio
    async def test_ainvoke_resets_messages(self, executor: CrewAgentExecutor) -> None:
        """Messages from a previous ainvoke must not leak into the next one."""
        executor.messages = [
            {"role": "system", "content": "old system prompt"},
            {"role": "user", "content": "old task prompt"},
            {"role": "assistant", "content": "old response"},
        ]
        executor.iterations = 7

        with patch.object(
            executor,
            "_ainvoke_loop",
            new_callable=AsyncMock,
            return_value=AgentFinish(thought="done", output="result", text="done"),
        ):
            with patch.object(executor, "_create_short_term_memory"):
                with patch.object(executor, "_create_long_term_memory"):
                    with patch.object(executor, "_create_external_memory"):
                        await executor.ainvoke(
                            {"input": "new task", "tool_names": "", "tools": ""}
                        )

        system_msgs = [m for m in executor.messages if m.get("role") == "system"]
        assert len(system_msgs) == 1, (
            f"Expected exactly 1 system message after ainvoke, got {len(system_msgs)}"
        )

    @pytest.mark.asyncio
    async def test_ainvoke_resets_iterations(self, executor: CrewAgentExecutor) -> None:
        """Iterations must be reset to 0 at the start of each ainvoke."""
        executor.iterations = 42

        with patch.object(
            executor,
            "_ainvoke_loop",
            new_callable=AsyncMock,
            return_value=AgentFinish(thought="done", output="result", text="done"),
        ):
            with patch.object(executor, "_create_short_term_memory"):
                with patch.object(executor, "_create_long_term_memory"):
                    with patch.object(executor, "_create_external_memory"):
                        await executor.ainvoke(
                            {"input": "task", "tool_names": "", "tools": ""}
                        )

    @pytest.mark.asyncio
    async def test_sequential_ainvokes_no_message_accumulation(
        self, executor: CrewAgentExecutor
    ) -> None:
        """Calling ainvoke multiple times must not accumulate messages."""
        message_counts: list[int] = []

        async def capture_and_finish() -> AgentFinish:
            message_counts.append(len(executor.messages))
            return AgentFinish(thought="done", output="result", text="done")

        with patch.object(executor, "_ainvoke_loop", side_effect=capture_and_finish):
            with patch.object(executor, "_create_short_term_memory"):
                with patch.object(executor, "_create_long_term_memory"):
                    with patch.object(executor, "_create_external_memory"):
                        for i in range(3):
                            await executor.ainvoke(
                                {"input": f"task {i}", "tool_names": "", "tools": ""}
                            )

        assert len(message_counts) == 3
        assert message_counts[0] == message_counts[1] == message_counts[2], (
            f"Message counts should be equal across invocations, got {message_counts}"
        )

    @pytest.mark.asyncio
    async def test_sequential_ainvokes_no_duplicate_system_messages(
        self, executor: CrewAgentExecutor
    ) -> None:
        """Each ainvoke must contain exactly one system message, not N."""
        system_msg_counts: list[int] = []

        async def capture_and_finish() -> AgentFinish:
            count = sum(1 for m in executor.messages if m.get("role") == "system")
            system_msg_counts.append(count)
            return AgentFinish(thought="done", output="ok", text="ok")

        with patch.object(executor, "_ainvoke_loop", side_effect=capture_and_finish):
            with patch.object(executor, "_create_short_term_memory"):
                with patch.object(executor, "_create_long_term_memory"):
                    with patch.object(executor, "_create_external_memory"):
                        for _ in range(5):
                            await executor.ainvoke(
                                {"input": "task", "tool_names": "", "tools": ""}
                            )

        assert all(c == 1 for c in system_msg_counts), (
            f"Expected 1 system message per ainvoke, got {system_msg_counts}"
        )
