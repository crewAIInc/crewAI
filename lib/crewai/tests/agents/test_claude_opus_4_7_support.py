"""Tests for Claude 4.7 Opus support (issue #5808).

Covers:
- BaseLLM.supports_assistant_prefill() default behaviour
- AnthropicCompletion.supports_assistant_prefill() model detection
- LLM (litellm) supports_assistant_prefill() detection
- Temperature parameter dropping for models that reject it
- CrewAgentExecutor message splitting for no-prefill models
- handle_max_iterations_exceeded prefill-aware message role
"""

from __future__ import annotations

from typing import Literal
from unittest.mock import MagicMock, patch

from crewai.llms.base_llm import BaseLLM
from crewai.utilities.agent_utils import format_message_for_llm


# ---------------------------------------------------------------------------
# BaseLLM.supports_assistant_prefill (default)
# ---------------------------------------------------------------------------


class TestBaseLLMPrefillDefault:
    """BaseLLM.supports_assistant_prefill() should default to True."""

    def test_base_llm_defaults_to_true(self):
        """The abstract base returns True so existing providers are
        unaffected unless they override."""
        llm = MagicMock(spec=BaseLLM)
        assert BaseLLM.supports_assistant_prefill(llm) is True


# ---------------------------------------------------------------------------
# AnthropicCompletion.supports_assistant_prefill
# ---------------------------------------------------------------------------


class TestAnthropicPrefillDetection:
    """AnthropicCompletion.supports_assistant_prefill() should return False
    for Claude 4.6+ models and True for earlier models."""

    def _make_anthropic_llm(self, model: str) -> object:
        from crewai.llms.providers.anthropic.completion import AnthropicCompletion

        llm = AnthropicCompletion.model_construct(model=model)
        return llm

    def test_claude_opus_4_7_no_prefill(self):
        llm = self._make_anthropic_llm("claude-opus-4-7")
        assert llm.supports_assistant_prefill() is False

    def test_claude_sonnet_4_6_no_prefill(self):
        llm = self._make_anthropic_llm("claude-sonnet-4-6")
        assert llm.supports_assistant_prefill() is False

    def test_claude_opus_4_5_supports_prefill(self):
        llm = self._make_anthropic_llm("claude-opus-4-5")
        assert llm.supports_assistant_prefill() is True

    def test_claude_3_5_sonnet_supports_prefill(self):
        llm = self._make_anthropic_llm("claude-3-5-sonnet-20241022")
        assert llm.supports_assistant_prefill() is True

    def test_claude_3_opus_supports_prefill(self):
        llm = self._make_anthropic_llm("claude-3-opus-20240229")
        assert llm.supports_assistant_prefill() is True

    def test_claude_5_0_no_prefill(self):
        """Future major version should also be detected."""
        llm = self._make_anthropic_llm("claude-5-0-opus")
        assert llm.supports_assistant_prefill() is False


# ---------------------------------------------------------------------------
# AnthropicCompletion temperature dropping
# ---------------------------------------------------------------------------


class TestAnthropicTemperatureDropping:
    """Claude 4.6+ models reject the temperature parameter."""

    def test_temperature_dropped_for_no_prefill_model(self):
        from crewai.llms.providers.anthropic.completion import AnthropicCompletion

        llm = AnthropicCompletion(
            model="claude-opus-4-7",
            max_tokens=4096,
            stream=False,
            temperature=0.7,
        )
        params = llm._prepare_completion_params(
            messages=[], system_message=None, tools=None
        )
        assert "temperature" not in params

    def test_temperature_kept_for_prefill_model(self):
        from crewai.llms.providers.anthropic.completion import AnthropicCompletion

        llm = AnthropicCompletion(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4096,
            stream=False,
            temperature=0.7,
        )
        params = llm._prepare_completion_params(
            messages=[], system_message=None, tools=None
        )
        assert params.get("temperature") == 0.7


# ---------------------------------------------------------------------------
# LLM (litellm) supports_assistant_prefill
# ---------------------------------------------------------------------------


class TestLiteLLMPrefillDetection:
    """LLM.supports_assistant_prefill() should use litellm.get_model_info()
    with a fallback to name-based heuristic.

    Since LLM.__new__ routes models to native provider subclasses, we
    test the method by invoking it as an unbound function on a plain
    object that has the necessary `.model` attribute.
    """

    def _call_method(self, model: str, **patches) -> bool:
        """Call LLM.supports_assistant_prefill on a lightweight stub."""
        from crewai.llm import LLM

        stub = MagicMock()
        stub.model = model
        return LLM.supports_assistant_prefill(stub)

    def test_litellm_detects_no_prefill_via_model_info(self):
        with patch("crewai.llm.litellm") as mock_litellm, \
             patch("crewai.llm.LITELLM_AVAILABLE", True):
            mock_litellm.get_model_info.return_value = {
                "litellm_provider": "anthropic",
                "supports_assistant_prefill": False,
            }
            assert self._call_method("claude-opus-4-7") is False

    def test_litellm_supports_prefill_for_older_claude(self):
        with patch("crewai.llm.litellm") as mock_litellm, \
             patch("crewai.llm.LITELLM_AVAILABLE", True):
            mock_litellm.get_model_info.return_value = {
                "litellm_provider": "anthropic",
                "supports_assistant_prefill": True,
            }
            assert self._call_method("claude-3-opus-20240229") is True

    def test_litellm_non_anthropic_defaults_to_true(self):
        with patch("crewai.llm.litellm") as mock_litellm, \
             patch("crewai.llm.LITELLM_AVAILABLE", True):
            mock_litellm.get_model_info.return_value = {
                "litellm_provider": "openai",
                "supports_assistant_prefill": False,
            }
            assert self._call_method("gpt-4o") is True

    def test_litellm_fallback_heuristic_claude_4_7(self):
        with patch("crewai.llm.litellm") as mock_litellm, \
             patch("crewai.llm.LITELLM_AVAILABLE", True):
            mock_litellm.get_model_info.side_effect = Exception("not found")
            assert self._call_method("claude-opus-4-7") is False

    def test_litellm_fallback_heuristic_non_claude(self):
        with patch("crewai.llm.litellm") as mock_litellm, \
             patch("crewai.llm.LITELLM_AVAILABLE", True):
            mock_litellm.get_model_info.side_effect = Exception("not found")
            assert self._call_method("some-custom-model") is True


# ---------------------------------------------------------------------------
# CrewAgentExecutor._append_message
# ---------------------------------------------------------------------------


class TestAppendAssistantResponse:
    """When the model does not support prefill, the observation part of the
    response must be split into a separate user-role message."""

    def _make_executor(self, supports_prefill: bool):
        from crewai.agents.crew_agent_executor import CrewAgentExecutor

        mock_llm = MagicMock()
        mock_llm.supports_stop_words.return_value = True
        mock_llm.supports_assistant_prefill.return_value = supports_prefill
        mock_llm.stop = None
        mock_llm.model = (
            "claude-opus-4-7" if not supports_prefill else "gpt-4o"
        )

        executor = CrewAgentExecutor.model_construct(
            llm=mock_llm,
            messages=[],
        )
        return executor

    def test_prefill_supported_single_assistant_message(self):
        executor = self._make_executor(supports_prefill=True)
        text = (
            "Thought: searching\n"
            "Action: search\n"
            "Action Input: query\n"
            "Observation: result"
        )
        executor._append_message(text)
        assert len(executor.messages) == 1
        assert executor.messages[0]["role"] == "assistant"

    def test_no_prefill_splits_observation_into_user_message(self):
        executor = self._make_executor(supports_prefill=False)
        text = (
            "Thought: searching\n"
            "Action: search\n"
            "Action Input: query\n"
            "Observation: result data"
        )
        executor._append_message(text)

        assert len(executor.messages) == 2
        assert executor.messages[0]["role"] == "assistant"
        assert "Observation" not in executor.messages[0]["content"]
        assert executor.messages[1]["role"] == "user"
        assert executor.messages[1]["content"].startswith("Observation:")

    def test_no_prefill_without_observation_adds_continuation(self):
        executor = self._make_executor(supports_prefill=False)
        text = "Thought: I must give my final answer\nFinal Answer: 42"
        executor._append_message(text)

        assert len(executor.messages) == 2
        assert executor.messages[0]["role"] == "assistant"
        assert executor.messages[1]["role"] == "user"

    def test_no_prefill_last_message_is_always_user(self):
        executor = self._make_executor(supports_prefill=False)

        # Case 1: with observation
        executor.messages = []
        executor._append_message(
            "Thought: x\nAction: y\nAction Input: z\nObservation: r"
        )
        assert executor.messages[-1]["role"] == "user"

        # Case 2: without observation
        executor.messages = []
        executor._append_message("Thought: done\nFinal Answer: 42")
        assert executor.messages[-1]["role"] == "user"

    def test_multiple_iterations_message_structure(self):
        executor = self._make_executor(supports_prefill=False)
        executor._append_message(
            "Thought: step 1\nAction: tool1\nAction Input: a\nObservation: res1"
        )
        executor._append_message(
            "Thought: step 2\nAction: tool2\nAction Input: b\nObservation: res2"
        )
        assert len(executor.messages) == 4
        roles = [m["role"] for m in executor.messages]
        assert roles == ["assistant", "user", "assistant", "user"]

    def test_user_role_messages_pass_through_unchanged(self):
        """Messages with role='user' should not be affected."""
        executor = self._make_executor(supports_prefill=False)
        executor._append_message("some user input", role="user")
        assert len(executor.messages) == 1
        assert executor.messages[0]["role"] == "user"

    def test_system_role_messages_pass_through_unchanged(self):
        executor = self._make_executor(supports_prefill=False)
        executor._append_message("system prompt", role="system")
        assert len(executor.messages) == 1
        assert executor.messages[0]["role"] == "system"

    def test_supports_prefill_property_graceful_fallback(self):
        """When the LLM doesn't have supports_assistant_prefill, default True."""
        from crewai.agents.crew_agent_executor import CrewAgentExecutor

        mock_llm = MagicMock(spec=[])  # Empty spec = no attributes
        executor = CrewAgentExecutor.model_construct(llm=mock_llm, messages=[])
        assert executor.supports_prefill is True


# ---------------------------------------------------------------------------
# handle_max_iterations_exceeded prefill-aware
# ---------------------------------------------------------------------------


class TestHandleMaxIterationsExceededPrefill:
    """handle_max_iterations_exceeded should use user role for the forced
    answer message when the model doesn't support prefill."""

    def test_no_prefill_uses_user_role(self):
        from crewai.utilities.agent_utils import handle_max_iterations_exceeded

        mock_llm = MagicMock()
        mock_llm.supports_assistant_prefill.return_value = False
        mock_llm.call.return_value = "Final Answer: done"

        messages: list[dict[str, str]] = []
        handle_max_iterations_exceeded(
            formatted_answer=None,
            printer=MagicMock(),
            messages=messages,
            llm=mock_llm,
            callbacks=[],
            verbose=False,
        )
        # The forced-answer message should be "user" role, not "assistant"
        assert any(m["role"] == "user" for m in messages)
        assert not any(
            m["role"] == "assistant" for m in messages
        ), "Should not have assistant message for no-prefill model"

    def test_prefill_uses_assistant_role(self):
        from crewai.utilities.agent_utils import handle_max_iterations_exceeded

        mock_llm = MagicMock()
        mock_llm.supports_assistant_prefill.return_value = True
        mock_llm.call.return_value = "Final Answer: done"

        messages: list[dict[str, str]] = []
        handle_max_iterations_exceeded(
            formatted_answer=None,
            printer=MagicMock(),
            messages=messages,
            llm=mock_llm,
            callbacks=[],
            verbose=False,
        )
        assert any(m["role"] == "assistant" for m in messages)
