"""Regression: LiteLLM emits a final usage-only chunk (choices=[]) when
``stream_options.include_usage`` is set. The old post-loop
``_extract_finish_reason_and_response_id(last_chunk)`` then silently returned
(None, None). These tests pin that we capture finish_reason/response_id
incrementally during the stream loop instead.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from crewai.events.event_bus import CrewAIEventsBus
from crewai.events.types.llm_events import LLMCallCompletedEvent
from crewai.llm import LLM


@pytest.fixture
def mock_emit():
    with patch.object(CrewAIEventsBus, "emit") as mock:
        yield mock


def _completed_event(mock_emit) -> LLMCallCompletedEvent:
    matches = [
        call.kwargs["event"]
        for call in mock_emit.call_args_list
        if isinstance(call.kwargs.get("event"), LLMCallCompletedEvent)
    ]
    assert matches, "expected an LLMCallCompletedEvent to be emitted"
    assert len(matches) == 1, f"expected one completed event, got {len(matches)}"
    return matches[0]


def _chunks_with_usage_tail() -> list[dict[str, Any]]:
    """Three-chunk stream mirroring LiteLLM's include_usage behavior:
    two content chunks where the second carries finish_reason="stop",
    then a final usage-only chunk with choices=[]."""
    return [
        {
            "id": "chatcmpl-stream-1",
            "choices": [
                {"delta": {"content": "hi"}, "finish_reason": None}
            ],
        },
        {
            "id": "chatcmpl-stream-1",
            "choices": [
                {"delta": {"content": " there"}, "finish_reason": "stop"}
            ],
        },
        {
            "id": "chatcmpl-stream-1",
            "choices": [],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "total_tokens": 3,
            },
        },
    ]


def test_sync_stream_emits_finish_reason_and_response_id_from_loop(mock_emit):
    llm = LLM(model="gpt-4o-mini", is_litellm=True, stream=True)

    with patch("crewai.llm.litellm.completion", return_value=iter(_chunks_with_usage_tail())):
        result = llm.call("anything")

    assert result == "hi there"

    event = _completed_event(mock_emit)
    assert event.finish_reason == "stop"
    assert event.response_id == "chatcmpl-stream-1"


@pytest.mark.asyncio
async def test_async_stream_emits_finish_reason_and_response_id_from_loop(mock_emit):
    llm = LLM(model="gpt-4o-mini", is_litellm=True, stream=True)

    async def _aiter():
        for chunk in _chunks_with_usage_tail():
            yield chunk

    async def _acompletion(*_args, **_kwargs):
        return _aiter()

    with patch("crewai.llm.litellm.acompletion", side_effect=_acompletion):
        result = await llm.acall("anything")

    assert result == "hi there"

    event = _completed_event(mock_emit)
    assert event.finish_reason == "stop"
    assert event.response_id == "chatcmpl-stream-1"
