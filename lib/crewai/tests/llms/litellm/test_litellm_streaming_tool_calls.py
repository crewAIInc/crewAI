"""Regression: LiteLLM streaming must rebuild tool-call deltas faithfully.

Streamed ``tool_calls`` are accumulated per ``index`` and rebuilt into
``ChatCompletionDeltaToolCall`` objects before being handed back to the
caller (the agent executor) or executed in-band. The reassembly dropped the
wire ``id`` — every rebuilt call carried ``id=None``, so the assistant and
tool messages the executor writes for the next LLM turn contain
``tool_calls[].id = None`` / ``tool_call_id = None``, which providers
reject — and, on the async path, fed the already-accumulated arguments back
into the per-chunk accumulator, doubling them (``{"a": 1}{"a": 1}``) so
``json.loads`` always failed and the tool call was silently dropped whenever
``available_functions`` was provided.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from crewai.llm import LLM


def _tool_call_chunks() -> list[Any]:
    """Two-chunk tool-call stream: the first delta carries the call ``id``
    and function name plus the first half of the JSON arguments; the second
    carries the rest of the arguments and ``finish_reason='tool_calls'``."""
    from litellm.types.utils import (
        ChatCompletionDeltaToolCall,
        Delta,
        Function,
        ModelResponseStream,
        StreamingChoices,
    )

    return [
        ModelResponseStream(
            id="chatcmpl-stream-1",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(
                        role="assistant",
                        tool_calls=[
                            ChatCompletionDeltaToolCall(
                                index=0,
                                id="call_abc123",
                                type="function",
                                function=Function(
                                    name="get_weather",
                                    arguments='{"city": "San',
                                ),
                            )
                        ],
                    ),
                    finish_reason=None,
                )
            ],
        ),
        ModelResponseStream(
            id="chatcmpl-stream-1",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(
                        tool_calls=[
                            ChatCompletionDeltaToolCall(
                                index=0,
                                function=Function(arguments=' Francisco"}'),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
        ),
    ]


def _weather_tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }


def test_sync_streaming_tool_calls_keep_wire_id():
    """Streamed tool calls returned to the executor must keep the wire id."""
    llm = LLM(model="gpt-4o-mini", is_litellm=True, stream=True)

    with patch(
        "crewai.llm.litellm.completion",
        return_value=iter(_tool_call_chunks()),
    ):
        result = llm.call("weather?", tools=[_weather_tool_schema()])

    assert isinstance(result, list) and len(result) == 1
    tool_call = result[0]
    assert tool_call.id == "call_abc123"
    assert tool_call.function.name == "get_weather"
    assert tool_call.function.arguments == '{"city": "San Francisco"}'


@pytest.mark.asyncio
async def test_async_streaming_tool_calls_keep_wire_id():
    """Same as the sync test: the executor correlates tool results by id."""
    llm = LLM(model="gpt-4o-mini", is_litellm=True, stream=True)

    async def _acompletion(*_args: Any, **_kwargs: Any) -> Any:
        async def _aiter():
            for chunk in _tool_call_chunks():
                yield chunk

        return _aiter()

    with patch("crewai.llm.litellm.acompletion", side_effect=_acompletion):
        result = await llm.acall("weather?", tools=[_weather_tool_schema()])

    assert isinstance(result, list) and len(result) == 1
    tool_call = result[0]
    assert tool_call.id == "call_abc123"
    assert tool_call.function.name == "get_weather"
    assert tool_call.function.arguments == '{"city": "San Francisco"}'


@pytest.mark.asyncio
async def test_async_streaming_executes_tool_calls_with_available_functions():
    """With ``available_functions`` the accumulated call must run once, not
    be re-accumulated (which doubles the arguments and never parses)."""
    llm = LLM(model="gpt-4o-mini", is_litellm=True, stream=True)
    get_weather = MagicMock(return_value="72F and sunny")

    async def _acompletion(*_args: Any, **_kwargs: Any) -> Any:
        async def _aiter():
            for chunk in _tool_call_chunks():
                yield chunk

        return _aiter()

    with patch("crewai.llm.litellm.acompletion", side_effect=_acompletion):
        result = await llm.acall(
            "weather?",
            tools=[_weather_tool_schema()],
            available_functions={"get_weather": get_weather},
        )

    get_weather.assert_called_once_with(city="San Francisco")
    assert result == "72F and sunny"
