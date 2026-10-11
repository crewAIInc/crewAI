"""Offline tests for #7013: OpenAI must warn when a response is truncated.

Covers the Chat Completions path (``finish_reason == "length"``), the
Responses API path (``status == "incomplete"`` resolved through
``incomplete_details.reason`` — non-token stops stay silent), the streaming
Responses path (``response.incomplete`` terminal events) and the structured
streaming finalize path (warning exactly once per response).
"""

import json
import logging
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pydantic
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


def test_responses_api_max_output_tokens_incomplete_warns_with_limit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``status="incomplete"`` warns only when ``incomplete_details.reason``
    is ``max_output_tokens``, and the warning carries the limit that was
    sent with the request (#7013)."""
    llm = OpenAICompletion(
        model="gpt-4o-mini", api_key="test-key", api="responses", max_tokens=50
    )
    response = MagicMock(
        id="resp_1",
        status="incomplete",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
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
    message = truncated[0].getMessage()
    assert "max_output_tokens" in message
    assert "token_limit=50" in message


def test_responses_api_incomplete_without_token_cap_is_silent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``incomplete`` alone is not a token cap: a ``content_filter`` stop
    must not produce the truncation warning."""
    llm = OpenAICompletion(
        model="gpt-4o-mini", api_key="test-key", api="responses"
    )
    response = MagicMock(
        id="resp_1",
        status="incomplete",
        incomplete_details=SimpleNamespace(reason="content_filter"),
        output=[],
        output_text="filtered answer",
        usage=MagicMock(
            input_tokens=10, output_tokens=5, total_tokens=15
        ),
    )
    client = MagicMock()
    client.responses.create.return_value = response
    llm._client = client

    with caplog.at_level(logging.WARNING):
        result = llm.call("Hello")

    assert result == "filtered answer"
    assert not [r for r in caplog.records if TRUNCATED in r.getMessage()]


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


def _responses_stream_events(
    terminal_status: str, incomplete_details: SimpleNamespace | None = None
) -> list[SimpleNamespace]:
    """Terminal-event stream for the Responses API: one delta, then the end."""
    return [
        SimpleNamespace(type="response.created", response=SimpleNamespace(id="resp_1")),
        SimpleNamespace(type="response.output_text.delta", delta="partial answer"),
        SimpleNamespace(
            type=f"response.{terminal_status}",
            response=SimpleNamespace(
                id="resp_1",
                status=terminal_status,
                incomplete_details=incomplete_details,
                output=[],
                output_text="partial answer",
                usage=SimpleNamespace(
                    input_tokens=10, output_tokens=5, total_tokens=15
                ),
            ),
        ),
    ]


def test_responses_stream_incomplete_warns_with_limit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A token-capped Responses stream ends with ``response.incomplete``:
    the warning must fire (it previously saw no terminal event at all)."""
    llm = OpenAICompletion(
        model="gpt-4o-mini", api_key="test-key", api="responses", stream=True,
        max_tokens=50,
    )
    client = MagicMock()
    client.responses.create.return_value = _responses_stream_events(
        "incomplete", SimpleNamespace(reason="max_output_tokens")
    )
    llm._client = client

    with caplog.at_level(logging.WARNING):
        result = llm.call("Hello")

    assert result == "partial answer"
    truncated = [r for r in caplog.records if TRUNCATED in r.getMessage()]
    assert len(truncated) == 1
    message = truncated[0].getMessage()
    assert "max_output_tokens" in message
    assert "token_limit=50" in message


def test_responses_stream_completed_is_silent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm = OpenAICompletion(
        model="gpt-4o-mini", api_key="test-key", api="responses", stream=True
    )
    client = MagicMock()
    client.responses.create.return_value = _responses_stream_events("completed")
    llm._client = client

    with caplog.at_level(logging.WARNING):
        result = llm.call("Hello")

    assert result == "partial answer"
    assert not [r for r in caplog.records if TRUNCATED in r.getMessage()]


@pytest.mark.asyncio
async def test_responses_stream_incomplete_async_warns_with_limit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Async mirror: ``_ahandle_streaming_responses`` must capture
    ``response.incomplete`` too."""
    llm = OpenAICompletion(
        model="gpt-4o-mini", api_key="test-key", api="responses", stream=True,
        max_tokens=50,
    )

    async def _event_stream() -> AsyncIterator[SimpleNamespace]:
        for event in _responses_stream_events(
            "incomplete", SimpleNamespace(reason="max_output_tokens")
        ):
            yield event

    client = MagicMock()
    client.responses.create = AsyncMock(return_value=_event_stream())
    llm._async_client = client

    with caplog.at_level(logging.WARNING):
        result = await llm.acall("Hello")

    assert result == "partial answer"
    truncated = [r for r in caplog.records if TRUNCATED in r.getMessage()]
    assert len(truncated) == 1
    message = truncated[0].getMessage()
    assert "max_output_tokens" in message
    assert "token_limit=50" in message


class _Answer(pydantic.BaseModel):
    answer: str


@pytest.mark.asyncio
async def test_async_structured_streaming_length_finish_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The async structured (``response_model``) streaming path must warn on
    a ``length`` finish too — it previously returned without calling
    ``warn_if_truncated``."""
    llm = _llm(stream=True)

    async def _chunk_stream() -> AsyncIterator[SimpleNamespace]:
        delta = SimpleNamespace(content="42", tool_calls=None)
        yield SimpleNamespace(
            id="chatcmpl-1",
            choices=[SimpleNamespace(delta=delta, finish_reason=None)],
            usage=None,
        )
        yield SimpleNamespace(
            id="chatcmpl-1",
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="", tool_calls=None),
                    finish_reason="length",
                )
            ],
            usage=None,
        )

    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_chunk_stream())
    llm._async_client = client
    params: dict[str, Any] = {
        "messages": [{"role": "user", "content": "Give me a number"}],
        "max_completion_tokens": 123,
    }

    with caplog.at_level(logging.WARNING):
        result = await llm._ahandle_streaming_completion(
            params, response_model=_Answer
        )

    assert result == "42"
    truncated = [r for r in caplog.records if TRUNCATED in r.getMessage()]
    assert len(truncated) == 1
    message = truncated[0].getMessage()
    assert "length" in message
    assert "token_limit=123" in message
