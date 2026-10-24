"""Offline tests for #7013: Gemini must warn when a response is truncated.

Gemini reports token-cap truncation as ``finishReason.name == "MAX_TOKENS"``.
"""

import logging
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

TRUNCATED = "truncated by token cap"


@pytest.fixture(autouse=True)
def mock_google_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    if (
        "GOOGLE_API_KEY" not in os.environ
        and "GEMINI_API_KEY" not in os.environ
    ):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")


def _chunk(
    text: str | None = None,
    finish_reason: str | None = None,
) -> SimpleNamespace:
    """Fake ``GenerateContentResponse``; the enum is carried by ``.name``."""
    parts = (
        [SimpleNamespace(text=text, function_call=None, thought=None)] if text else []
    )
    candidate = SimpleNamespace(
        content=SimpleNamespace(parts=parts) if parts else None,
        finish_reason=SimpleNamespace(name=finish_reason) if finish_reason else None,
    )
    return SimpleNamespace(
        candidates=[candidate],
        usage_metadata=SimpleNamespace(
            prompt_token_count=10,
            candidates_token_count=5,
            total_token_count=15,
            cached_content_token_count=0,
        ),
        response_id=None,
    )


def _gemini_llm(**kwargs: object) -> object:
    from crewai.llms.providers.gemini.completion import GeminiCompletion

    return GeminiCompletion(model="gemini-2.0-flash-001", **kwargs)


def test_non_streaming_max_tokens_warns(caplog: pytest.LogCaptureFixture) -> None:
    llm = _gemini_llm()
    client = MagicMock()
    client.models.generate_content.return_value = _chunk(
        text="partial answer", finish_reason="MAX_TOKENS"
    )
    llm._client = client

    with caplog.at_level(logging.WARNING):
        result = llm.call("Hello")

    assert result == "partial answer"  # return value semantics unchanged
    truncated = [r for r in caplog.records if TRUNCATED in r.getMessage()]
    assert len(truncated) == 1
    assert "gemini" in truncated[0].getMessage()
    assert "MAX_TOKENS" in truncated[0].getMessage()


def test_non_streaming_stop_is_silent(caplog: pytest.LogCaptureFixture) -> None:
    llm = _gemini_llm()
    client = MagicMock()
    client.models.generate_content.return_value = _chunk(
        text="a complete answer", finish_reason="STOP"
    )
    llm._client = client

    with caplog.at_level(logging.WARNING):
        result = llm.call("Hello")

    assert result == "a complete answer"
    assert not [r for r in caplog.records if TRUNCATED in r.getMessage()]


def test_streaming_max_tokens_warns_exactly_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm = _gemini_llm(stream=True)
    client = MagicMock()
    client.models.generate_content_stream.return_value = iter(
        [
            _chunk(text="partial answer"),
            _chunk(finish_reason="MAX_TOKENS"),
        ]
    )
    llm._client = client

    with caplog.at_level(logging.WARNING):
        result = llm.call("Hello")

    assert result == "partial answer"
    truncated = [r for r in caplog.records if TRUNCATED in r.getMessage()]
    assert len(truncated) == 1
    assert "MAX_TOKENS" in truncated[0].getMessage()


def test_streaming_stop_is_silent(caplog: pytest.LogCaptureFixture) -> None:
    llm = _gemini_llm(stream=True)
    client = MagicMock()
    client.models.generate_content_stream.return_value = iter(
        [
            _chunk(text="a complete answer"),
            _chunk(finish_reason="STOP"),
        ]
    )
    llm._client = client

    with caplog.at_level(logging.WARNING):
        result = llm.call("Hello")

    assert result == "a complete answer"
    assert not [r for r in caplog.records if TRUNCATED in r.getMessage()]
