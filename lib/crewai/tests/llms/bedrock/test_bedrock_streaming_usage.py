"""Streamed Bedrock calls report the usage from the trailing metadata event.

Converse sends usage in a ``metadata`` event after ``messageStop``. A stream
loop that stops at ``messageStop`` or returns at ``contentBlockStop`` never
reads it, so the call reports no tokens.
"""

import os
from typing import Any
from unittest.mock import MagicMock, patch

from pydantic import BaseModel
import pytest

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.llm_events import LLMCallCompletedEvent
from crewai.llm import LLM
from crewai.llms.providers.bedrock.completion import (
    STRUCTURED_OUTPUT_TOOL_NAME,
    BedrockCompletion,
)


USAGE = {"inputTokens": 50, "outputTokens": 25, "totalTokens": 75}


class City(BaseModel):
    name: str


def _text_stream() -> list[dict[str, Any]]:
    return [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": "Paris"}, "contentBlockIndex": 0}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "end_turn"}},
        {"metadata": {"usage": USAGE}},
    ]


def _structured_stream() -> list[dict[str, Any]]:
    return [
        {"messageStart": {"role": "assistant"}},
        {
            "contentBlockStart": {
                "start": {
                    "toolUse": {
                        "toolUseId": "tool-1",
                        "name": STRUCTURED_OUTPUT_TOOL_NAME,
                    }
                },
                "contentBlockIndex": 0,
            }
        },
        {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"name": "Paris"}'}}}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "tool_use"}},
        {"metadata": {"usage": USAGE}},
    ]


class _AsyncStream:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._it = iter(events)

    def __aiter__(self) -> "_AsyncStream":
        return self

    async def __anext__(self) -> dict[str, Any]:
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration from None


def _build_completion() -> BedrockCompletion:
    with patch.dict(
        os.environ,
        {
            "AWS_ACCESS_KEY_ID": "test-access-key",
            "AWS_SECRET_ACCESS_KEY": "test-secret-key",
            "AWS_DEFAULT_REGION": "us-east-1",
        },
    ):
        with patch("crewai.llms.providers.bedrock.completion.Session"):
            llm = LLM(model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0")
    assert isinstance(llm, BedrockCompletion)
    return llm


@pytest.mark.parametrize(
    "events, response_model",
    [(_text_stream(), None), (_structured_stream(), City)],
    ids=["text", "response_model"],
)
def test_sync_stream_reports_usage(
    events: list[dict[str, Any]], response_model: type[BaseModel] | None
) -> None:
    llm = _build_completion()
    client = MagicMock()
    client.converse_stream.return_value = {"stream": events}
    completed: list[LLMCallCompletedEvent] = []

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(LLMCallCompletedEvent)
        def on_completed(source: Any, event: LLMCallCompletedEvent) -> None:
            completed.append(event)

        with patch.object(llm, "_get_sync_client", return_value=client):
            llm._handle_streaming_converse(
                messages=[{"role": "user", "content": "capital of France?"}],
                body={},
                response_model=response_model,
            )
        crewai_event_bus.flush(timeout=10.0)

    assert llm.get_token_usage_summary().total_tokens == 75
    assert [e.usage for e in completed] == [USAGE]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "events, response_model",
    [(_text_stream(), None), (_structured_stream(), City)],
    ids=["text", "response_model"],
)
async def test_async_stream_reports_usage(
    events: list[dict[str, Any]], response_model: type[BaseModel] | None
) -> None:
    llm = _build_completion()

    async def converse_stream(**kwargs: Any) -> dict[str, Any]:
        return {"stream": _AsyncStream(events)}

    client = MagicMock()
    client.converse_stream = converse_stream

    async def ensure_client() -> MagicMock:
        return client

    completed: list[LLMCallCompletedEvent] = []

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(LLMCallCompletedEvent)
        def on_completed(source: Any, event: LLMCallCompletedEvent) -> None:
            completed.append(event)

        with patch.object(llm, "_ensure_async_client", side_effect=ensure_client):
            await llm._ahandle_streaming_converse(
                messages=[{"role": "user", "content": "capital of France?"}],
                body={},
                response_model=response_model,
            )
        crewai_event_bus.flush(timeout=10.0)

    assert llm.get_token_usage_summary().total_tokens == 75
    assert [e.usage for e in completed] == [USAGE]


def test_tool_block_after_structured_output_does_not_run() -> None:
    """The stream now continues past the structured output block, so a tool
    block that follows it in the same message must still be skipped."""
    events = _structured_stream()
    events[4:4] = [
        {
            "contentBlockStart": {
                "start": {"toolUse": {"toolUseId": "tool-2", "name": "get_weather"}},
                "contentBlockIndex": 1,
            }
        },
        {"contentBlockDelta": {"delta": {"toolUse": {"input": "{}"}}}},
        {"contentBlockStop": {"contentBlockIndex": 1}},
    ]
    llm = _build_completion()
    client = MagicMock()
    client.converse_stream.return_value = {"stream": events}
    calls: list[dict[str, Any]] = []

    def get_weather(**kwargs: Any) -> str:
        calls.append(kwargs)
        return "sunny"

    with patch.object(llm, "_get_sync_client", return_value=client):
        result = llm._handle_streaming_converse(
            messages=[{"role": "user", "content": "capital of France?"}],
            body={},
            available_functions={"get_weather": get_weather},
            response_model=City,
        )

    assert result == City(name="Paris")
    assert calls == []
    assert llm.get_token_usage_summary().total_tokens == 75
