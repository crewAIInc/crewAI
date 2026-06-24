"""Regression tests for SyncHumanInputProvider result display (#6072)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import crewai.core.providers.human_input as human_input_module
from crewai.agents.parser import AgentFinish
from crewai.core.providers.human_input import SyncHumanInputProvider
from crewai.events.event_listener import event_listener


class _FakeAgent:
    def __init__(self, verbose: bool) -> None:
        self.verbose = verbose
        self.role = "Researcher"


class _FakeCrew:
    def __init__(self, verbose: bool) -> None:
        self.verbose = verbose
        self._train = False


class _FakeContext:
    """Minimal ExecutorContext stand-in for the accept-on-first-round path."""

    def __init__(self, agent: _FakeAgent, crew: _FakeCrew) -> None:
        self.agent = agent
        self.crew = crew
        self.ask_for_human_input = True
        self.messages: list = []

    def _is_training_mode(self) -> bool:
        return False


def _answer() -> AgentFinish:
    return AgentFinish(thought="", output="THE FINAL RESULT", text="THE FINAL RESULT")


class TestResultDisplayBeforeFeedback:
    def test_result_shown_before_prompt_when_not_verbose(self) -> None:
        """Non-verbose + human_input must render the result before prompting,
        otherwise the operator is asked to approve output they never saw.
        """
        provider = SyncHumanInputProvider()
        context = _FakeContext(_FakeAgent(verbose=False), _FakeCrew(verbose=False))
        answer = _answer()

        formatter = MagicMock()
        with (
            patch.object(event_listener, "formatter", formatter),
            patch("builtins.input", return_value=""),
        ):
            result = provider.handle_feedback(answer, context)  # type: ignore[arg-type]

        formatter.handle_agent_logs_execution.assert_called_once_with(
            "Researcher", answer, verbose=True
        )
        # The result panel must be rendered BEFORE the feedback prompt panel.
        method_calls = [c[0] for c in formatter.mock_calls]
        render_idx = method_calls.index("handle_agent_logs_execution")
        prompt_idx = method_calls.index("console.print")
        assert render_idx < prompt_idx
        assert result is answer

    def test_result_not_reshown_when_verbose(self) -> None:
        """When verbose, the executor already rendered the result, so the
        provider must not re-render it (avoid a duplicate panel).
        """
        provider = SyncHumanInputProvider()
        context = _FakeContext(_FakeAgent(verbose=False), _FakeCrew(verbose=True))

        formatter = MagicMock()
        with (
            patch.object(event_listener, "formatter", formatter),
            patch("builtins.input", return_value=""),
        ):
            provider.handle_feedback(_answer(), context)  # type: ignore[arg-type]

        formatter.handle_agent_logs_execution.assert_not_called()


class TestResultDisplayBeforeFeedbackAsync:
    """The async feedback path must show the result before prompting too."""

    @pytest.mark.asyncio
    async def test_result_shown_before_prompt_when_not_verbose(self) -> None:
        provider = SyncHumanInputProvider()
        context = _FakeContext(_FakeAgent(verbose=False), _FakeCrew(verbose=False))
        answer = _answer()

        formatter = MagicMock()
        with (
            patch.object(event_listener, "formatter", formatter),
            patch.object(
                human_input_module, "_async_readline", new=AsyncMock(return_value="")
            ),
        ):
            result = await provider.handle_feedback_async(answer, context)  # type: ignore[arg-type]

        formatter.handle_agent_logs_execution.assert_called_once_with(
            "Researcher", answer, verbose=True
        )
        method_calls = [c[0] for c in formatter.mock_calls]
        render_idx = method_calls.index("handle_agent_logs_execution")
        prompt_idx = method_calls.index("console.print")
        assert render_idx < prompt_idx
        assert result is answer

    @pytest.mark.asyncio
    async def test_result_not_reshown_when_verbose(self) -> None:
        provider = SyncHumanInputProvider()
        context = _FakeContext(_FakeAgent(verbose=False), _FakeCrew(verbose=True))

        formatter = MagicMock()
        with (
            patch.object(event_listener, "formatter", formatter),
            patch.object(
                human_input_module, "_async_readline", new=AsyncMock(return_value="")
            ),
        ):
            await provider.handle_feedback_async(_answer(), context)  # type: ignore[arg-type]

        formatter.handle_agent_logs_execution.assert_not_called()
