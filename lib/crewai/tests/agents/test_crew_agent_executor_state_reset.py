"""Tests for CrewAgentExecutor state reset between task executions.

Verifies that messages and iterations are properly cleared when the same
executor instance is reused across sequential tasks, preventing context
pollution and premature max-iteration exits.

Related issues: #4319, #4389, #4661
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from crewai.agents.crew_agent_executor import CrewAgentExecutor
from crewai.agents.parser import AgentFinish


@pytest.fixture
def _make_executor() -> callable:
    """Factory fixture that builds a CrewAgentExecutor with mocked deps."""

    def _factory(**overrides: Any) -> CrewAgentExecutor:
        llm = Mock()
        llm.supports_stop_words.return_value = True
        llm.stop = []
        llm.supports_function_calling.return_value = False

        task = Mock()
        task.id = "task-1"
        task.description = "Test task"
        task.human_input = False
        task.response_model = None
        task.output_pydantic = None
        task.output_json = None

        crew = MagicMock()
        crew.id = "crew-1"
        crew._memory = None
        crew._train = False
        crew.verbose = False

        agent = MagicMock()
        agent.id = "agent-1"
        agent.role = "Tester"
        agent.verbose = False
        agent.key = "tester-key"
        agent.security_config = MagicMock()
        agent.security_config.fingerprint = "fp-123"

        tools_handler = Mock()
        tools_handler.cache = None

        defaults: dict[str, Any] = {
            "llm": llm,
            "task": task,
            "crew": crew,
            "agent": agent,
            "prompt": {"system": "You are a helper.", "user": "Do {input}"},
            "max_iter": 10,
            "tools": [],
            "tools_names": "",
            "stop_words": ["Observation:"],
            "tools_description": "",
            "tools_handler": tools_handler,
        }
        defaults.update(overrides)
        return CrewAgentExecutor(**defaults)

    return _factory


class TestCrewAgentExecutorStateReset:
    """Ensure invoke() and ainvoke() reset execution state."""

    # ------------------------------------------------------------------
    # Synchronous invoke
    # ------------------------------------------------------------------

    @patch(
        "crewai.agents.crew_agent_executor.get_all_files",
        return_value=None,
    )
    @patch(
        "crewai.agents.crew_agent_executor.get_llm_response",
        return_value="Final Answer: done",
    )
    def test_invoke_resets_messages_between_calls(
        self,
        _mock_llm_response: Mock,
        _mock_files: Mock,
        _make_executor: callable,
    ) -> None:
        """Messages list should be fresh on every invoke() call."""
        executor = _make_executor()

        inputs: dict[str, str] = {"input": "task-one", "tool_names": "", "tools": ""}

        # First invocation
        executor.invoke(inputs)

        # Messages should have been populated during the first run
        msgs_after_first = list(executor.messages)
        assert len(msgs_after_first) > 0

        # Second invocation (simulating reuse for a different task)
        executor.invoke(inputs)

        # The messages list should NOT contain leftovers from the first run.
        # It should start fresh with only messages from the second invocation.
        # Specifically, there should be exactly one system and one user message
        # from _setup_messages, plus any assistant messages from the loop.
        system_msgs = [m for m in executor.messages if m.get("role") == "system"]
        assert len(system_msgs) == 1, (
            f"Expected exactly 1 system message after second invoke, "
            f"got {len(system_msgs)}"
        )

    @patch(
        "crewai.agents.crew_agent_executor.get_all_files",
        return_value=None,
    )
    @patch(
        "crewai.agents.crew_agent_executor.get_llm_response",
        return_value="Final Answer: done",
    )
    def test_invoke_resets_iterations_between_calls(
        self,
        _mock_llm_response: Mock,
        _mock_files: Mock,
        _make_executor: callable,
    ) -> None:
        """Iterations counter should reset to 0 on every invoke() call."""
        executor = _make_executor()

        inputs: dict[str, str] = {"input": "task-one", "tool_names": "", "tools": ""}

        executor.invoke(inputs)
        assert executor.iterations > 0, "iterations should have incremented"

        # Second invocation
        executor.invoke(inputs)

        # iterations should have been reset and only reflect the second run
        # (for a single-pass answer it should be 1)
        assert executor.iterations == 1, (
            f"Expected iterations == 1 after reset, got {executor.iterations}"
        )

    # ------------------------------------------------------------------
    # Asynchronous ainvoke
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    @patch(
        "crewai.agents.crew_agent_executor.aget_all_files",
        return_value=None,
    )
    @patch(
        "crewai.agents.crew_agent_executor.aget_llm_response",
        return_value="Final Answer: done",
    )
    async def test_ainvoke_resets_messages_between_calls(
        self,
        _mock_llm_response: Mock,
        _mock_files: Mock,
        _make_executor: callable,
    ) -> None:
        """Messages list should be fresh on every ainvoke() call."""
        executor = _make_executor()

        inputs: dict[str, str] = {"input": "task-one", "tool_names": "", "tools": ""}

        await executor.ainvoke(inputs)

        msgs_after_first = list(executor.messages)
        assert len(msgs_after_first) > 0

        await executor.ainvoke(inputs)

        system_msgs = [m for m in executor.messages if m.get("role") == "system"]
        assert len(system_msgs) == 1, (
            f"Expected exactly 1 system message after second ainvoke, "
            f"got {len(system_msgs)}"
        )

    @pytest.mark.asyncio
    @patch(
        "crewai.agents.crew_agent_executor.aget_all_files",
        return_value=None,
    )
    @patch(
        "crewai.agents.crew_agent_executor.aget_llm_response",
        return_value="Final Answer: done",
    )
    async def test_ainvoke_resets_iterations_between_calls(
        self,
        _mock_llm_response: Mock,
        _mock_files: Mock,
        _make_executor: callable,
    ) -> None:
        """Iterations counter should reset to 0 on every ainvoke() call."""
        executor = _make_executor()

        inputs: dict[str, str] = {"input": "task-one", "tool_names": "", "tools": ""}

        await executor.ainvoke(inputs)
        assert executor.iterations > 0

        await executor.ainvoke(inputs)

        assert executor.iterations == 1, (
            f"Expected iterations == 1 after reset, got {executor.iterations}"
        )

    # ------------------------------------------------------------------
    # Regression: multiple sequential tasks via the same executor
    # ------------------------------------------------------------------

    @patch(
        "crewai.agents.crew_agent_executor.get_all_files",
        return_value=None,
    )
    @patch(
        "crewai.agents.crew_agent_executor.get_llm_response",
        return_value="Final Answer: result",
    )
    def test_no_context_leak_across_three_sequential_invokes(
        self,
        _mock_llm_response: Mock,
        _mock_files: Mock,
        _make_executor: callable,
    ) -> None:
        """Simulates an agent reused across 3 sequential tasks.

        After each invoke the message count should be consistent and not
        grow unboundedly.
        """
        executor = _make_executor()
        inputs: dict[str, str] = {"input": "task", "tool_names": "", "tools": ""}

        message_counts: list[int] = []
        for _ in range(3):
            executor.invoke(inputs)
            message_counts.append(len(executor.messages))

        # All three runs should produce the same number of messages
        assert message_counts[0] == message_counts[1] == message_counts[2], (
            f"Message counts diverged across sequential invokes: {message_counts}"
        )
