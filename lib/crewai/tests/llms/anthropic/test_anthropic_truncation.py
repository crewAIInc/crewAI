"""Offline tests for #7013: Anthropic must warn when a response is truncated.

Anthropic reports token-cap truncation as ``stop_reason == "max_tokens"``.
"""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from crewai.llms.providers.anthropic.completion import AnthropicCompletion

TRUNCATED = "truncated by token cap"


def _message(stop_reason: str | None, text: str = "partial answer") -> MagicMock:
    return MagicMock(
        id="msg_1",
        stop_reason=stop_reason,
        content=[MagicMock(text=text)],
        usage=MagicMock(
            input_tokens=10,
            output_tokens=5,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
    )


class _SyncStream:
    """Minimal double for the Anthropic SDK streaming context manager."""

    def __init__(self, events: list[SimpleNamespace], final_message: MagicMock):
        self._events = events
        self._final_message = final_message

    def __enter__(self) -> "_SyncStream":
        return self

    def __exit__(self, *_args: object) -> bool:
        return False

    def __iter__(self) -> object:
        return iter(self._events)

    def get_final_message(self) -> MagicMock:
        return self._final_message


def _llm(**kwargs: object) -> AnthropicCompletion:
    return AnthropicCompletion(model="claude-fable-5", api_key="test-key", **kwargs)


def _text_delta_events(text: str = "partial answer") -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            type="content_block_delta",
            delta=SimpleNamespace(type="text_delta", text=text),
        ),
    ]


def test_non_streaming_max_tokens_warns(caplog: pytest.LogCaptureFixture) -> None:
    llm = _llm()
    client = MagicMock()
    client.messages.create.return_value = _message("max_tokens")
    llm._client = client

    with caplog.at_level(logging.WARNING):
        result = llm.call("Hello")

    assert result == "partial answer"  # return value semantics unchanged
    truncated = [r for r in caplog.records if TRUNCATED in r.getMessage()]
    assert len(truncated) == 1
    assert "anthropic" in truncated[0].getMessage()
    assert "max_tokens" in truncated[0].getMessage()


def test_non_streaming_end_turn_is_silent(caplog: pytest.LogCaptureFixture) -> None:
    llm = _llm()
    client = MagicMock()
    client.messages.create.return_value = _message("end_turn", text="a complete answer")
    llm._client = client

    with caplog.at_level(logging.WARNING):
        result = llm.call("Hello")

    assert result == "a complete answer"
    assert not [r for r in caplog.records if TRUNCATED in r.getMessage()]


def test_streaming_max_tokens_warns_exactly_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm = _llm(stream=True)
    client = MagicMock()
    client.messages.stream.return_value = _SyncStream(
        _text_delta_events(), _message("max_tokens")
    )
    llm._client = client

    with caplog.at_level(logging.WARNING):
        result = llm.call("Hello")

    assert result == "partial answer"
    truncated = [r for r in caplog.records if TRUNCATED in r.getMessage()]
    assert len(truncated) == 1


def test_streaming_end_turn_is_silent(caplog: pytest.LogCaptureFixture) -> None:
    llm = _llm(stream=True)
    client = MagicMock()
    client.messages.stream.return_value = _SyncStream(
        _text_delta_events("a complete answer"),
        _message("end_turn", text="a complete answer"),
    )
    llm._client = client

    with caplog.at_level(logging.WARNING):
        result = llm.call("Hello")

    assert result == "a complete answer"
    assert not [r for r in caplog.records if TRUNCATED in r.getMessage()]
