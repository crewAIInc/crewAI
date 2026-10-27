"""Regression: on the LiteLLM fallback path, two kinds of call reached the
platform without token usage, so traces showed requests with 0 tokens/$0.

- Native tool-call turns (``available_functions`` is None, as the agent
  executor calls it) returned the tool calls without emitting
  ``LLMCallCompletedEvent`` at all.
- Structured output through Instructor (``response_model``) emitted the event
  with ``usage=None``.

Native providers already emit a ``tool_call`` completion with usage; these
tests pin the same for LiteLLM.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from litellm import ModelResponse
from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    Function,
    ModelResponseStream,
)
from pydantic import BaseModel

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.llm_events import LLMCallCompletedEvent, LLMCallType
from crewai.llm import LLM

USAGE = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    }
]


class Brief(BaseModel):
    title: str


@pytest.fixture
def mock_emit():
    with patch.object(crewai_event_bus, "emit") as mock:
        yield mock


def _completed_event(mock_emit) -> LLMCallCompletedEvent:
    matches = [
        call.kwargs["event"]
        for call in mock_emit.call_args_list
        if isinstance(call.kwargs.get("event"), LLMCallCompletedEvent)
    ]
    assert len(matches) == 1, f"expected one completed event, got {len(matches)}"
    return matches[0]


def _tool_call_response(name: str, arguments: str) -> ModelResponse:
    return ModelResponse(
        id="chatcmpl-tools-1",
        choices=[
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        }
                    ],
                },
            }
        ],
        usage=USAGE,
    )


def _tool_call_stream() -> list[Any]:
    return [
        {
            "id": "chatcmpl-stream-tools",
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            ChatCompletionDeltaToolCall(
                                index=0,
                                id="call_1",
                                function=Function(
                                    name="get_weather",
                                    arguments='{"location": "NYC"}',
                                ),
                            )
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        },
        # LiteLLM's include_usage tail, as the real stream yields it.
        ModelResponseStream(id="chatcmpl-stream-tools", choices=[], usage=USAGE),
    ]


def _assert_tool_call_usage(event: LLMCallCompletedEvent) -> None:
    assert event.call_type == LLMCallType.TOOL_CALL
    assert event.usage is not None
    assert event.usage["prompt_tokens"] == 11
    assert event.usage["completion_tokens"] == 7


def test_tool_call_turn_emits_completed_event_with_usage(mock_emit):
    llm = LLM(model="gpt-4o-mini", is_litellm=True)

    with patch(
        "crewai.llm.litellm.completion",
        return_value=_tool_call_response("get_weather", '{"location": "NYC"}'),
    ):
        result = llm.call("What's the weather in NYC?", tools=TOOLS)

    assert result[0].function.name == "get_weather"
    event = _completed_event(mock_emit)
    _assert_tool_call_usage(event)
    assert event.finish_reason == "tool_calls"
    assert event.response_id == "chatcmpl-tools-1"


@pytest.mark.asyncio
async def test_async_tool_call_turn_emits_completed_event_with_usage(mock_emit):
    llm = LLM(model="gpt-4o-mini", is_litellm=True)

    with patch(
        "crewai.llm.litellm.acompletion",
        new=AsyncMock(
            return_value=_tool_call_response("get_weather", '{"location": "NYC"}')
        ),
    ):
        result = await llm.acall("What's the weather in NYC?", tools=TOOLS)

    assert result[0].function.name == "get_weather"
    _assert_tool_call_usage(_completed_event(mock_emit))


def test_streaming_tool_call_turn_emits_completed_event_with_usage(mock_emit):
    llm = LLM(model="gpt-4o-mini", is_litellm=True, stream=True)

    with patch("crewai.llm.litellm.completion", return_value=iter(_tool_call_stream())):
        result = llm.call("What's the weather in NYC?", tools=TOOLS)

    assert result[0].function.name == "get_weather"
    event = _completed_event(mock_emit)
    _assert_tool_call_usage(event)
    assert event.response_id == "chatcmpl-stream-tools"


@pytest.mark.asyncio
async def test_async_streaming_tool_call_turn_emits_completed_event_with_usage(
    mock_emit,
):
    llm = LLM(model="gpt-4o-mini", is_litellm=True, stream=True)

    async def _aiter():
        for chunk in _tool_call_stream():
            yield chunk

    async def _acompletion(*_args, **_kwargs):
        return _aiter()

    with patch("crewai.llm.litellm.acompletion", side_effect=_acompletion):
        result = await llm.acall("What's the weather in NYC?", tools=TOOLS)

    assert result[0].function.name == "get_weather"
    _assert_tool_call_usage(_completed_event(mock_emit))


def test_structured_output_emits_completed_event_with_usage(mock_emit):
    llm = LLM(model="gpt-4o-mini", is_litellm=True)

    with patch(
        "litellm.completion",
        return_value=_tool_call_response("Brief", '{"title": "Usage matters"}'),
    ):
        result = llm.call("Write a brief", response_model=Brief)

    assert Brief.model_validate_json(result).title == "Usage matters"
    event = _completed_event(mock_emit)
    assert event.call_type == LLMCallType.LLM_CALL
    assert event.usage is not None
    assert event.usage["prompt_tokens"] == 11
    assert event.usage["completion_tokens"] == 7
    assert llm.get_token_usage_summary().total_tokens == 18
