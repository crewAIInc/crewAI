"""A response cut off by the token cap is otherwise indistinguishable from a
complete one: the partial text is returned as a successful result. Only the
Bedrock provider used to warn about it, so these tests pin the warning down at
the shared emission point every provider funnels through.
"""

from __future__ import annotations

import logging
import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from crewai.events.types.llm_events import LLMCallType
from crewai.llm import LLM
from crewai.llms._finish_reason_utils import is_truncation_finish_reason
from crewai.llms.base_llm import BaseLLM


class _StubLLM(BaseLLM):
    model: str = "test-model"

    def call(self, *args: Any, **kwargs: Any) -> str:
        return ""

    async def acall(self, *args: Any, **kwargs: Any) -> str:
        return ""

    def supports_function_calling(self) -> bool:
        return False


def _truncation_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.WARNING and "truncated" in record.getMessage()
    ]


class TestIsTruncationFinishReason:
    @pytest.mark.parametrize(
        "finish_reason",
        [
            "length",  # OpenAI, Azure, LiteLLM, openai_compatible, Snowflake
            "max_tokens",  # Anthropic, Bedrock
            "MAX_TOKENS",  # Gemini
            "max_output_tokens",
            "max_completion_tokens",
            "Length",
        ],
    )
    def test_recognises_provider_truncation_vocabulary(self, finish_reason: str) -> None:
        assert is_truncation_finish_reason(finish_reason)

    @pytest.mark.parametrize(
        "finish_reason",
        [
            "stop",
            "end_turn",
            "STOP",
            "tool_calls",
            "tool_use",
            "content_filter",
            "content_filtered",
            "completed",
            "",
            None,
            42,
            MagicMock(),
        ],
    )
    def test_ignores_everything_else(self, finish_reason: Any) -> None:
        assert not is_truncation_finish_reason(finish_reason)


class TestWarningAtSharedEmissionPoint:
    # Every native provider reports completion through
    # BaseLLM._emit_call_completed_event, so covering it covers all of them.
    @pytest.mark.parametrize(
        "finish_reason", ["length", "max_tokens", "MAX_TOKENS", "max_output_tokens"]
    )
    def test_warns_once_naming_the_cap(
        self, finish_reason: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        llm = _StubLLM(model="test-model", max_tokens=16)

        with caplog.at_level(logging.WARNING):
            llm._emit_call_completed_event(
                response="partial",
                call_type=LLMCallType.LLM_CALL,
                finish_reason=finish_reason,
            )

        warnings = _truncation_warnings(caplog)
        assert len(warnings) == 1
        assert finish_reason in warnings[0]
        assert "max_tokens" in warnings[0]
        assert "16" in warnings[0]

    @pytest.mark.parametrize("finish_reason", ["stop", "tool_calls", None])
    def test_stays_silent_for_complete_responses(
        self, finish_reason: str | None, caplog: pytest.LogCaptureFixture
    ) -> None:
        llm = _StubLLM(model="test-model", max_tokens=16)

        with caplog.at_level(logging.WARNING):
            llm._emit_call_completed_event(
                response="all of it",
                call_type=LLMCallType.LLM_CALL,
                finish_reason=finish_reason,
            )

        assert _truncation_warnings(caplog) == []

    def test_reports_the_provider_specific_cap_field(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # OpenAI caps generation through max_completion_tokens, so the message
        # has to name that value rather than the unset max_tokens.
        llm = LLM(model="gpt-4o", max_completion_tokens=512)
        assert llm.max_tokens is None

        with caplog.at_level(logging.WARNING):
            llm._emit_call_completed_event(
                response="partial",
                call_type=LLMCallType.LLM_CALL,
                finish_reason="length",
            )

        warnings = _truncation_warnings(caplog)
        assert len(warnings) == 1
        assert "512" in warnings[0]

    def test_warns_on_empty_response_that_spent_the_whole_budget(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Reasoning models can burn the cap on thinking and emit no text; the
        # finish reason is the only thing separating that from a refusal.
        llm = _StubLLM(model="test-model", max_tokens=200)

        with caplog.at_level(logging.WARNING):
            llm._emit_call_completed_event(
                response="",
                call_type=LLMCallType.LLM_CALL,
                finish_reason="length",
            )

        assert len(_truncation_warnings(caplog)) == 1


class TestLiteLLMTruncationWarning:
    def _model_response(self, finish_reason: str) -> Any:
        from litellm.types.utils import Choices, Message, ModelResponse

        return ModelResponse(
            id="chatcmpl-truncated",
            choices=[
                Choices(
                    finish_reason=finish_reason,
                    index=0,
                    message=Message(content="The TCP three-way handshake is", role="assistant"),
                )
            ],
        )

    def test_warns_on_truncated_completion(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        llm = LLM(model="gpt-4o-mini", is_litellm=True, max_tokens=16)

        with (
            caplog.at_level(logging.WARNING),
            patch(
                "crewai.llm.litellm.completion",
                return_value=self._model_response("length"),
            ),
        ):
            llm.call("Explain the TCP three-way handshake in detail.")

        warnings = _truncation_warnings(caplog)
        assert len(warnings) == 1
        assert "16" in warnings[0]

    def test_stays_silent_on_complete_completion(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        llm = LLM(model="gpt-4o-mini", is_litellm=True, max_tokens=16)

        with (
            caplog.at_level(logging.WARNING),
            patch(
                "crewai.llm.litellm.completion",
                return_value=self._model_response("stop"),
            ),
        ):
            llm.call("Explain the TCP three-way handshake in detail.")

        assert _truncation_warnings(caplog) == []


class TestBedrockTruncationWarning:
    # Bedrock warned on its own before the shared check existed; the provider
    # check was dropped so a truncated Bedrock call still warns exactly once.
    def _converse_response(self, stop_reason: str) -> dict[str, Any]:
        return {
            "stopReason": stop_reason,
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "The TCP three-way handshake is"}],
                }
            },
            "usage": {"inputTokens": 10, "outputTokens": 16, "totalTokens": 26},
        }

    @pytest.fixture
    def bedrock_llm(self):
        with (
            patch.dict(
                os.environ,
                {
                    "AWS_ACCESS_KEY_ID": "test-access-key",
                    "AWS_SECRET_ACCESS_KEY": "test-secret-key",
                    "AWS_DEFAULT_REGION": "us-east-1",
                },
            ),
            patch(
                "crewai.llms.providers.bedrock.completion.Session"
            ) as mock_session_class,
        ):
            mock_client = MagicMock()
            mock_session_instance = MagicMock()
            mock_session_instance.client.return_value = mock_client
            mock_session_class.return_value = mock_session_instance

            llm = LLM(
                model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
                max_tokens=16,
            )
            yield llm, mock_client

    def test_warns_exactly_once(
        self, bedrock_llm, caplog: pytest.LogCaptureFixture
    ) -> None:
        llm, mock_client = bedrock_llm
        mock_client.converse.return_value = self._converse_response("max_tokens")

        with caplog.at_level(logging.WARNING):
            llm.call("Explain the TCP three-way handshake in detail.")

        assert len(_truncation_warnings(caplog)) == 1

    def test_stays_silent_on_complete_response(
        self, bedrock_llm, caplog: pytest.LogCaptureFixture
    ) -> None:
        llm, mock_client = bedrock_llm
        mock_client.converse.return_value = self._converse_response("end_turn")

        with caplog.at_level(logging.WARNING):
            llm.call("Explain the TCP three-way handshake in detail.")

        assert _truncation_warnings(caplog) == []
