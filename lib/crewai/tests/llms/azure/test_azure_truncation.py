"""Offline tests for #7013: Azure must warn when a response is truncated.

Azure AI Inference is choices-shaped: token-cap truncation arrives as
``finish_reason == "length"``.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest
from azure.ai.inference.models import StreamingChatCompletionsUpdate

from crewai.llms.providers.azure.completion import AzureCompletion

TRUNCATED = "truncated by token cap"


def _llm(**kwargs: object) -> AzureCompletion:
    return AzureCompletion(
        model="gpt-4o",
        api_key="test-key",
        endpoint="https://test.openai.azure.com",
        **kwargs,
    )


def _chat_completions(finish_reason: str | None, content: str) -> MagicMock:
    return MagicMock(
        id="chatcmpl-1",
        choices=[
            MagicMock(
                finish_reason=finish_reason,
                message=MagicMock(content=content, tool_calls=None),
            )
        ],
        usage=MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )


def _chunks(finish_reason: str | None) -> list[StreamingChatCompletionsUpdate]:
    return [
        StreamingChatCompletionsUpdate(
            id="chatcmpl-1",
            model="gpt-4o",
            created=None,
            choices=[
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "partial answer"},
                    "finish_reason": None,
                }
            ],
        ),
        StreamingChatCompletionsUpdate(
            id="chatcmpl-1",
            model="gpt-4o",
            created=None,
            choices=[
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": None},
                    "finish_reason": finish_reason,
                }
            ],
        ),
    ]


def test_non_streaming_length_finish_reason_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm = _llm()

    with caplog.at_level(logging.WARNING):
        with patch.object(llm._client, "complete") as mock_complete:
            mock_complete.return_value = _chat_completions("length", "partial answer")
            result = llm.call("Hello")

    assert result == "partial answer"  # return value semantics unchanged
    truncated = [r for r in caplog.records if TRUNCATED in r.getMessage()]
    assert len(truncated) == 1
    assert "azure" in truncated[0].getMessage()
    assert "length" in truncated[0].getMessage()


def test_non_streaming_stop_finish_reason_is_silent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm = _llm()

    with caplog.at_level(logging.WARNING):
        with patch.object(llm._client, "complete") as mock_complete:
            mock_complete.return_value = _chat_completions(
                "stop", "a complete answer"
            )
            result = llm.call("Hello")

    assert result == "a complete answer"
    assert not [r for r in caplog.records if TRUNCATED in r.getMessage()]


def test_streaming_length_finish_reason_warns_exactly_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm = _llm(stream=True)

    with caplog.at_level(logging.WARNING):
        with patch.object(llm._client, "complete") as mock_complete:
            mock_complete.return_value = iter(_chunks("length"))
            result = llm.call("Hello")

    assert result == "partial answer"
    truncated = [r for r in caplog.records if TRUNCATED in r.getMessage()]
    assert len(truncated) == 1
    assert "length" in truncated[0].getMessage()


def test_streaming_stop_finish_reason_is_silent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm = _llm(stream=True)

    with caplog.at_level(logging.WARNING):
        with patch.object(llm._client, "complete") as mock_complete:
            mock_complete.return_value = iter(_chunks("stop"))
            result = llm.call("Hello")

    assert result == "partial answer"
    assert not [r for r in caplog.records if TRUNCATED in r.getMessage()]
