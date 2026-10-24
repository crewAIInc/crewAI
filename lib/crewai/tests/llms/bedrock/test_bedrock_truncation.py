"""Offline tests for #7013: Bedrock's truncation warning via the shared util.

Bedrock already warned on ``stopReason == "max_tokens"`` (issue #7013's
premise); these tests pin the migrated shared-util path — same behavior,
one normalized message across providers.
"""

import logging
import os
from unittest.mock import MagicMock, patch

import pytest

from crewai.llm import LLM

TRUNCATED = "truncated by token cap"


@pytest.fixture(autouse=True)
def mock_aws_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    if not (
        os.environ.get("AWS_ACCESS_KEY_ID")
        and os.environ.get("AWS_SECRET_ACCESS_KEY")
    ):
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access-key")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-key")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


def _bedrock_llm(stream: bool = False) -> tuple[LLM, MagicMock]:
    with patch("crewai.llms.providers.bedrock.completion.Session") as session_cls:
        client = MagicMock()
        session_instance = MagicMock()
        session_instance.client.return_value = client
        session_cls.return_value = session_instance
        llm = LLM(model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0", stream=stream)
    return llm, client


def _converse_response(stop_reason: str | None) -> dict[str, object]:
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "partial answer"}],
            }
        },
        "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
        "stopReason": stop_reason,
    }


def _stream_events(stop_reason: str | None) -> dict[str, object]:
    return {
        "stream": [
            {"contentBlockDelta": {"delta": {"text": "partial answer"}}},
            {"messageStop": {"stopReason": stop_reason}},
        ]
    }


def test_non_streaming_max_tokens_warns_via_shared_util(
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm, client = _bedrock_llm()

    with caplog.at_level(logging.WARNING):
        with patch.object(client, "converse") as mock_converse:
            mock_converse.return_value = _converse_response("max_tokens")
            result = llm.call("Hello")

    assert result == "partial answer"  # return value semantics unchanged
    truncated = [r for r in caplog.records if TRUNCATED in r.getMessage()]
    assert len(truncated) == 1
    assert "bedrock" in truncated[0].getMessage()
    assert "max_tokens" in truncated[0].getMessage()


def test_non_streaming_end_turn_is_silent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm, client = _bedrock_llm()

    with caplog.at_level(logging.WARNING):
        with patch.object(client, "converse") as mock_converse:
            mock_converse.return_value = _converse_response("end_turn")
            result = llm.call("Hello")

    assert result == "partial answer"
    assert not [r for r in caplog.records if TRUNCATED in r.getMessage()]


def test_streaming_max_tokens_warns_exactly_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm, client = _bedrock_llm(stream=True)

    with caplog.at_level(logging.WARNING):
        with patch.object(client, "converse_stream") as mock_stream:
            mock_stream.return_value = _stream_events("max_tokens")
            result = llm.call("Hello")

    assert result == "partial answer"
    truncated = [r for r in caplog.records if TRUNCATED in r.getMessage()]
    assert len(truncated) == 1
    assert "max_tokens" in truncated[0].getMessage()


def test_streaming_end_turn_is_silent(caplog: pytest.LogCaptureFixture) -> None:
    llm, client = _bedrock_llm(stream=True)

    with caplog.at_level(logging.WARNING):
        with patch.object(client, "converse_stream") as mock_stream:
            mock_stream.return_value = _stream_events("end_turn")
            result = llm.call("Hello")

    assert result == "partial answer"
    assert not [r for r in caplog.records if TRUNCATED in r.getMessage()]
