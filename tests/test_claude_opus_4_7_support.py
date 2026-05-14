"""Tests for Claude 4.7 Opus support (issue #5808).

Covers:
- LLM.supports_assistant_prefill() detection
- Temperature parameter dropping for models that reject it
- CrewAgentExecutor message splitting for no-prefill models
"""

from unittest.mock import MagicMock, patch

import pytest

from crewai.llm import LLM

# ---------------------------------------------------------------------------
# LLM.supports_assistant_prefill
# ---------------------------------------------------------------------------


class TestSupportsAssistantPrefill:
    """LLM.supports_assistant_prefill() should return False for Anthropic
    models that do not support prefill and True otherwise."""

    @patch("crewai.llm.litellm.get_model_info")
    def test_claude_opus_4_7_no_prefill(self, mock_info):
        mock_info.return_value = {
            "litellm_provider": "anthropic",
            "supports_assistant_prefill": False,
        }
        llm = LLM(model="claude-opus-4-7")
        assert llm.supports_assistant_prefill() is False

    @patch("crewai.llm.litellm.get_model_info")
    def test_claude_3_opus_supports_prefill(self, mock_info):
        mock_info.return_value = {
            "litellm_provider": "anthropic",
            "supports_assistant_prefill": True,
        }
        llm = LLM(model="claude-3-opus-20240229")
        assert llm.supports_assistant_prefill() is True

    @patch("crewai.llm.litellm.get_model_info")
    def test_openai_model_supports_prefill(self, mock_info):
        """Non-Anthropic models should default to True even when the field
        is False in model info (the flag only matters for Anthropic)."""
        mock_info.return_value = {
            "litellm_provider": "openai",
            "supports_assistant_prefill": False,
        }
        llm = LLM(model="gpt-4o")
        assert llm.supports_assistant_prefill() is True

    @patch("crewai.llm.litellm.get_model_info")
    def test_anthropic_provider_prefix(self, mock_info):
        mock_info.return_value = {
            "litellm_provider": "anthropic",
            "supports_assistant_prefill": False,
        }
        llm = LLM(model="anthropic/claude-opus-4-7")
        assert llm.supports_assistant_prefill() is False

    @patch("crewai.llm.litellm.get_model_info")
    def test_fallback_heuristic_claude_4_7(self, mock_info):
        """When litellm cannot resolve the model, the name-based heuristic
        should detect Claude 4.6+ patterns."""
        mock_info.side_effect = Exception("model not found")
        llm = LLM(model="claude-opus-4-7")
        assert llm.supports_assistant_prefill() is False

    @patch("crewai.llm.litellm.get_model_info")
    def test_fallback_heuristic_claude_4_6(self, mock_info):
        mock_info.side_effect = Exception("model not found")
        llm = LLM(model="claude-sonnet-4-6")
        assert llm.supports_assistant_prefill() is False

    @patch("crewai.llm.litellm.get_model_info")
    def test_fallback_heuristic_claude_5(self, mock_info):
        mock_info.side_effect = Exception("model not found")
        llm = LLM(model="claude-5-0-opus")
        assert llm.supports_assistant_prefill() is False

    @patch("crewai.llm.litellm.get_model_info")
    def test_fallback_heuristic_claude_3_5(self, mock_info):
        """Claude 3.5 should still support prefill."""
        mock_info.side_effect = Exception("model not found")
        llm = LLM(model="claude-3-5-sonnet-20241022")
        assert llm.supports_assistant_prefill() is True

    @patch("crewai.llm.litellm.get_model_info")
    def test_fallback_non_claude_model(self, mock_info):
        mock_info.side_effect = Exception("model not found")
        llm = LLM(model="some-custom-model")
        assert llm.supports_assistant_prefill() is True


# ---------------------------------------------------------------------------
# Temperature dropping for no-prefill Anthropic models
# ---------------------------------------------------------------------------


class TestTemperatureDropping:
    """Claude 4.6+ models reject the temperature parameter.  The LLM.call()
    method should strip it before forwarding to litellm."""

    @patch("crewai.llm.litellm.completion")
    def test_temperature_dropped_for_claude_4_7(self, mock_completion):
        mock_completion.return_value = {
            "choices": [{"message": {"content": "response"}}]
        }
        llm = LLM(model="claude-opus-4-7", temperature=0.7)
        with patch.object(
            llm, "_is_anthropic_no_prefill_model", return_value=True
        ):
            llm.call([{"role": "user", "content": "hi"}])

        call_kwargs = mock_completion.call_args
        assert "temperature" not in call_kwargs.kwargs and "temperature" not in (
            call_kwargs.args[0] if call_kwargs.args else {}
        )
        # Check the actual keyword arguments passed
        passed_params = call_kwargs[1] if call_kwargs[1] else {}
        assert "temperature" not in passed_params

    @patch("crewai.llm.litellm.completion")
    def test_temperature_kept_for_normal_models(self, mock_completion):
        mock_completion.return_value = {
            "choices": [{"message": {"content": "response"}}]
        }
        llm = LLM(model="gpt-4o", temperature=0.7)
        with patch.object(
            llm, "_is_anthropic_no_prefill_model", return_value=False
        ):
            llm.call([{"role": "user", "content": "hi"}])

        passed_params = mock_completion.call_args[1]
        assert passed_params.get("temperature") == 0.7


# ---------------------------------------------------------------------------
# CrewAgentExecutor._append_assistant_response
# ---------------------------------------------------------------------------


class TestAppendAssistantResponse:
    """When the model does not support prefill, the observation part of the
    response must be split into a separate user-role message."""

    def _make_executor(self, supports_prefill: bool):
        """Build a minimal CrewAgentExecutor with the prefill flag set."""
        from crewai.agents.crew_agent_executor import CrewAgentExecutor

        # Build a mock LLM that returns the desired prefill support
        mock_llm = MagicMock()
        mock_llm.supports_stop_words.return_value = True
        mock_llm.supports_assistant_prefill.return_value = supports_prefill
        mock_llm.stop = None
        mock_llm.model = (
            "claude-opus-4-7" if not supports_prefill else "gpt-4o"
        )

        mock_agent = MagicMock()
        mock_agent.id = "test-agent"

        executor = CrewAgentExecutor(
            llm=mock_llm,
            task=MagicMock(),
            crew=MagicMock(),
            agent=mock_agent,
            prompt={"system": "sys", "user": "usr"},
            max_iter=5,
            tools=[],
            tools_names="",
            stop_words=["\nObservation:"],
            tools_description="",
            tools_handler=MagicMock(),
        )
        return executor

    def test_prefill_supported_single_assistant_message(self):
        """When prefill IS supported, the text should be added as a single
        assistant message (existing behaviour)."""
        executor = self._make_executor(supports_prefill=True)
        text = (
            "Thought: searching\n"
            "Action: search\n"
            "Action Input: query\n"
            "Observation: result"
        )
        executor._append_assistant_response(text)
        assert len(executor.messages) == 1
        assert executor.messages[0]["role"] == "assistant"
        assert executor.messages[0]["content"] == text.rstrip()

    def test_no_prefill_splits_observation_into_user_message(self):
        """When prefill is NOT supported, the observation should become a
        separate user message so the conversation does not end with an
        assistant turn."""
        executor = self._make_executor(supports_prefill=False)
        text = (
            "Thought: searching\n"
            "Action: search\n"
            "Action Input: query\n"
            "Observation: result data"
        )
        executor._append_assistant_response(text)

        assert len(executor.messages) == 2
        assert executor.messages[0]["role"] == "assistant"
        assert "Observation" not in executor.messages[0]["content"]
        assert executor.messages[1]["role"] == "user"
        assert executor.messages[1]["content"].startswith("Observation:")

    def test_no_prefill_without_observation_adds_continuation(self):
        """When there is no Observation marker (e.g. forced answer scenario),
        a generic user continuation message should be appended."""
        executor = self._make_executor(supports_prefill=False)
        text = "Thought: I must give my final answer\nFinal Answer: 42"
        executor._append_assistant_response(text)

        assert len(executor.messages) == 2
        assert executor.messages[0]["role"] == "assistant"
        assert executor.messages[1]["role"] == "user"

    def test_no_prefill_with_force_answer_and_observation(self):
        """When force-answer text is appended after the observation, the split
        should put everything from Observation: onward into the user message."""
        executor = self._make_executor(supports_prefill=False)
        text = (
            "Thought: searching\n"
            "Action: search\n"
            "Action Input: query\n"
            "Observation: tool result\n"
            "Now it's time you MUST give your absolute best final answer."
        )
        executor._append_assistant_response(text)

        assert len(executor.messages) == 2
        assert executor.messages[0]["role"] == "assistant"
        assert executor.messages[1]["role"] == "user"
        assert "tool result" in executor.messages[1]["content"]
        assert "MUST give" in executor.messages[1]["content"]

    def test_no_prefill_last_message_is_always_user(self):
        """Regardless of message content, the last message must always be
        from the user role when prefill is not supported."""
        executor = self._make_executor(supports_prefill=False)

        # Case 1: with observation
        executor.messages = []
        executor._append_assistant_response(
            "Thought: x\nAction: y\nAction Input: z\nObservation: r"
        )
        assert executor.messages[-1]["role"] == "user"

        # Case 2: without observation
        executor.messages = []
        executor._append_assistant_response("Thought: done\nFinal Answer: 42")
        assert executor.messages[-1]["role"] == "user"

    def test_multiple_iterations_message_structure(self):
        """Simulate multiple tool-use iterations and verify the message
        structure stays valid for no-prefill models."""
        executor = self._make_executor(supports_prefill=False)

        # First iteration
        executor._append_assistant_response(
            "Thought: step 1\nAction: tool1\nAction Input: a\nObservation: res1"
        )
        # Second iteration
        executor._append_assistant_response(
            "Thought: step 2\nAction: tool2\nAction Input: b\nObservation: res2"
        )

        assert len(executor.messages) == 4  # 2 assistant + 2 user
        # Verify alternation: assistant, user, assistant, user
        roles = [m["role"] for m in executor.messages]
        assert roles == ["assistant", "user", "assistant", "user"]
