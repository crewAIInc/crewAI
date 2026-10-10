"""Offline tests for #7013: OpenAI must warn when a response is truncated.

Covers the Chat Completions path (``finish_reason == "length"``), the
Responses API path (``status == "incomplete"``), and the streaming finalize
path (warning exactly once per response).
"""

import json
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from crewai.llms.providers.openai.completion import OpenAICompletion

TRUNCATED = "truncated by token cap"


def _llm(**kwargs: object) -> OpenAICompletion:
    return OpenAICompletion(model="gpt-4o-mini", api_key="test-key", **kwargs)


def _raw_double(finish_reason: str | None, content: str = "partial answer") -> MagicMock:
    """Double for ``chat.completions.with_raw_response.create``.

    ``text`` must be real JSON carrying ``choices`` so the gateway
    error-envelope guard defers to ``parse()``.
    """
    parsed = MagicMock(
        id="chatcmpl-1",
        choices=[
            MagicMock(
                finish_reason=finish_reason,
                message=MagicMock(content=content, tool_calls=None),
            )
        ],
        usage=MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )
    return MagicMock(
        text=json.dumps({"choices": [{"index": 0}]}),
        **{"parse.return_value": parsed},
    )


def _chat_chunks(finish_reason: str | None) -> list[SimpleNamespace]:
    """Two-chunk stream: one content delta, then the terminal chunk."""
    delta = SimpleNamespace(content="partial answer", tool_calls=None)
    return [
        SimpleNamespace(
            id="chatcmpl-1",
            choices=[SimpleNamespace(delta=delta, finish_reason=None)],
            usage=None,
        ),
        SimpleNamespace(
            id="chatcmpl-1",
            choices=[SimpleNamespace(delta=SimpleNamespace(content="", tool_calls=None), finish_reason=finish_reason)],
            usage=None,
        ),
    ]


def test_chat_completions_length_finish_reason_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm = _llm()
    client = MagicMock()
    client.chat.completions.with_raw_response.create.return_value = _raw_double(
        "length"
    )
    llm._client = client

    with caplog.at_level(logging.WARNING):
        result = llm.call("Hello")

    assert result == "partial answer"  # return value semantics unchanged
    truncated = [r for r in caplog.records if TRUNCATED in r.getMessage()]
    assert len(truncated) == 1
    assert "openai" in truncated[0].getMessage()
    assert "length" in truncated[0].getMessage()


def test_chat_completions_stop_finish_reason_is_silent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm = _llm()
    client = MagicMock()
    client.chat.completions.with_raw_response.create.return_value = _raw_double(
        "stop", content="a complete answer"
    )
    llm._client = client

    with caplog.at_level(logging.WARNING):
        result = llm.call("Hello")

    assert result == "a complete answer"
    assert not [r for r in caplog.records if TRUNCATED in r.getMessage()]


def test_streaming_length_finish_reason_warns_exactly_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm = _llm(stream=True)
    client = MagicMock()
    client.chat.completions.create.return_value = iter(_chat_chunks("length"))
    llm._client = client

    with caplog.at_level(logging.WARNING):
        result = llm.call("Hello")

    assert result == "partial answer"
    truncated = [r for r in caplog.records if TRUNCATED in r.getMessage()]
    assert len(truncated) == 1
    assert "length" in truncated[0].getMessage()


def test_streaming_stop_finish_reason_is_silent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm = _llm(stream=True)
    client = MagicMock()
    client.chat.completions.create.return_value = iter(_chat_chunks("stop"))
    llm._client = client

    with caplog.at_level(logging.WARNING):
        result = llm.call("Hello")

    assert result == "partial answer"
    assert not [r for r in caplog.records if TRUNCATED in r.getMessage()]


def test_responses_api_incomplete_status_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The Responses API reports ``status`` rather than a finish reason."""
    llm = OpenAICompletion(
        model="gpt-4o-mini", api_key="test-key", api="responses"
    )
    response = MagicMock(
        id="resp_1",
        status="incomplete",
        output=[],
        output_text="partial answer",
        usage=MagicMock(
            input_tokens=10, output_tokens=5, total_tokens=15
        ),
    )
    client = MagicMock()
    client.responses.create.return_value = response
    llm._client = client

    with caplog.at_level(logging.WARNING):
        result = llm.call("Hello")

    assert result == "partial answer"
    truncated = [r for r in caplog.records if TRUNCATED in r.getMessage()]
    assert len(truncated) == 1
    assert "incomplete" in truncated[0].getMessage()


def test_responses_api_completed_status_is_silent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm = OpenAICompletion(
        model="gpt-4o-mini", api_key="test-key", api="responses"
    )
    response = MagicMock(
        id="resp_1",
        status="completed",
        output=[],
        output_text="a complete answer",
        usage=MagicMock(
            input_tokens=10, output_tokens=5, total_tokens=15
        ),
    )
    client = MagicMock()
    client.responses.create.return_value = response
    llm._client = client

    with caplog.at_level(logging.WARNING):
        result = llm.call("Hello")

    assert result == "a complete answer"
    assert not [r for r in caplog.records if TRUNCATED in r.getMessage()]
