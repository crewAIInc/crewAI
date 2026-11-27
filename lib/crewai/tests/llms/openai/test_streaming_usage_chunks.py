"""Streaming chunks that carry ``usage`` must not lose their ``choices``.

OpenAI-compatible servers may put the token usage on the same chunk as the
last text delta, the last tool-call argument fragment and/or the
``finish_reason`` (DeepSeek documents exactly that for its terminal chunk).
The Chat Completions streaming handlers used to ``continue`` as soon as a
chunk had usage, silently dropping those choices. See issue #8775.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from openai.types.chat import ChatCompletionChunk
from pydantic import BaseModel

from crewai.llms.providers.openai.completion import OpenAICompletion


USAGE = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
PARAMS: dict[str, Any] = {"messages": [{"role": "user", "content": "hi"}]}

pytestmark = pytest.mark.asyncio


def make_chunk(
    content: str | None = None,
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str | None = None,
    usage: dict[str, int] | None = None,
    no_choices: bool = False,
) -> ChatCompletionChunk:
    delta: dict[str, Any] = {}
    if content is not None:
        delta["content"] = content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    choices = (
        []
        if no_choices
        else [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
    )
    return ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "gpt-4o-mini",
            "choices": choices,
            "usage": usage,
        }
    )


def tool_delta(
    index: int,
    arguments: str,
    *,
    call_id: str | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    function: dict[str, Any] = {"arguments": arguments}
    if name is not None:
        function["name"] = name
    delta: dict[str, Any] = {"index": index, "type": "function", "function": function}
    if call_id is not None:
        delta["id"] = call_id
    return delta


class _AsyncChunks:
    def __init__(self, chunks: Iterable[ChatCompletionChunk]) -> None:
        self._it = iter(list(chunks))

    def __aiter__(self) -> _AsyncChunks:
        return self

    async def __anext__(self) -> ChatCompletionChunk:
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration from None


def _sync_client(chunks: list[ChatCompletionChunk]) -> SimpleNamespace:
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_: iter(list(chunks)))
        )
    )


def _async_client(chunks: list[ChatCompletionChunk]) -> SimpleNamespace:
    async def create(**_: Any) -> _AsyncChunks:
        return _AsyncChunks(chunks)

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


async def run_stream(
    mode: str,
    chunks: list[ChatCompletionChunk],
    *,
    response_model: type[BaseModel] | None = None,
    available_functions: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any], list[str], OpenAICompletion]:
    """Feed ``chunks`` through the sync or async streaming handler.

    Returns (result, completed-event kwargs, streamed text chunks, llm).
    """
    llm = OpenAICompletion(model="gpt-4o-mini", api_key="sk-unit-test-key-0001", stream=True)
    streamed: list[str] = []

    def record_chunk(*_: Any, **kwargs: Any) -> None:
        if kwargs.get("call_type") is None and kwargs.get("chunk"):
            streamed.append(kwargs["chunk"])

    with (
        patch.object(OpenAICompletion, "_get_sync_client", return_value=_sync_client(chunks)),
        patch.object(OpenAICompletion, "_get_async_client", return_value=_async_client(chunks)),
        patch.object(OpenAICompletion, "_emit_stream_chunk_event", side_effect=record_chunk),
        patch.object(OpenAICompletion, "_emit_call_completed_event") as completed,
    ):
        if mode == "sync":
            result = llm._handle_streaming_completion(
                dict(PARAMS), available_functions=available_functions
            )
        else:
            result = await llm._ahandle_streaming_completion(
                dict(PARAMS),
                available_functions=available_functions,
                response_model=response_model,
            )
    event = completed.call_args.kwargs if completed.call_args else {}
    return result, event, streamed, llm


def assert_usage_counted_once(llm: OpenAICompletion) -> None:
    assert llm._token_usage["prompt_tokens"] == USAGE["prompt_tokens"]
    assert llm._token_usage["completion_tokens"] == USAGE["completion_tokens"]
    assert llm._token_usage["total_tokens"] == USAGE["total_tokens"]


MODES = ["sync", "async"]


@pytest.mark.parametrize("mode", MODES)
class TestTextStreaming:
    async def test_final_text_delta_with_usage_is_kept(self, mode: str) -> None:
        chunks = [
            make_chunk("HELLO-"),
            make_chunk("END", finish_reason="stop", usage=USAGE),
        ]
        result, event, streamed, llm = await run_stream(mode, chunks)
        assert result == "HELLO-END"
        assert event["finish_reason"] == "stop"
        assert event["usage"]["total_tokens"] == USAGE["total_tokens"]
        assert streamed == ["HELLO-", "END"]
        assert_usage_counted_once(llm)

    async def test_single_chunk_with_all_text_and_usage(self, mode: str) -> None:
        chunks = [make_chunk("HELLO-END", finish_reason="stop", usage=USAGE)]
        result, event, streamed, llm = await run_stream(mode, chunks)
        assert result == "HELLO-END"
        assert event["finish_reason"] == "stop"
        assert streamed == ["HELLO-END"]
        assert_usage_counted_once(llm)

    async def test_terminal_empty_delta_keeps_finish_reason(self, mode: str) -> None:
        # DeepSeek's documented terminal chunk: empty delta + finish_reason + usage.
        chunks = [
            make_chunk("HELLO-END"),
            make_chunk("", finish_reason="stop", usage=USAGE),
        ]
        result, event, _, llm = await run_stream(mode, chunks)
        assert result == "HELLO-END"
        assert event["finish_reason"] == "stop"
        assert_usage_counted_once(llm)

    async def test_length_finish_reason_with_usage_is_kept(self, mode: str) -> None:
        chunks = [
            make_chunk("truncated "),
            make_chunk("answer", finish_reason="length", usage=USAGE),
        ]
        result, event, _, _ = await run_stream(mode, chunks)
        assert result == "truncated answer"
        assert event["finish_reason"] == "length"

    async def test_usage_only_chunk_without_choices_still_works(self, mode: str) -> None:
        # OpenAI's own shape with stream_options.include_usage: a trailing
        # chunk with choices=[] that only carries usage.
        chunks = [
            make_chunk("HELLO-"),
            make_chunk("END", finish_reason="stop"),
            make_chunk(usage=USAGE, no_choices=True),
        ]
        result, event, streamed, llm = await run_stream(mode, chunks)
        assert result == "HELLO-END"
        assert event["finish_reason"] == "stop"
        assert streamed == ["HELLO-", "END"]
        assert_usage_counted_once(llm)

    async def test_stream_without_usage(self, mode: str) -> None:
        chunks = [make_chunk("HELLO-"), make_chunk("END", finish_reason="stop")]
        result, event, _, llm = await run_stream(mode, chunks)
        assert result == "HELLO-END"
        assert event["finish_reason"] == "stop"
        assert event["usage"] is None
        assert llm._token_usage["total_tokens"] == 0

    async def test_usage_on_a_middle_chunk_does_not_drop_text(self, mode: str) -> None:
        chunks = [
            make_chunk("A"),
            make_chunk("B", usage=USAGE),
            make_chunk("C", finish_reason="stop"),
        ]
        result, event, _, llm = await run_stream(mode, chunks)
        assert result == "ABC"
        assert event["finish_reason"] == "stop"
        assert_usage_counted_once(llm)

    @pytest.mark.parametrize("seed", range(25))
    async def test_random_chunking_round_trips(self, mode: str, seed: int) -> None:
        rng = random.Random(seed)
        text = "".join(
            rng.choice("abcdefghij klmnopqrstuvwxyz.,\n東京ü🙂") for _ in range(300)
        )
        cuts = sorted(rng.sample(range(1, len(text)), rng.randint(0, 40)))
        pieces = [text[i:j] for i, j in zip([0, *cuts], [*cuts, len(text)])]
        chunks = [make_chunk(p) for p in pieces[:-1]]
        chunks.append(make_chunk(pieces[-1], finish_reason="stop", usage=USAGE))
        if rng.random() < 0.5:
            # sometimes also send OpenAI's separate usage-only trailer
            chunks.append(make_chunk(usage=USAGE, no_choices=True))
        result, event, streamed, llm = await run_stream(mode, chunks)
        assert result == text
        assert "".join(streamed) == text
        assert event["finish_reason"] == "stop"
        assert_usage_counted_once(llm)


@pytest.mark.parametrize("mode", MODES)
class TestToolCallStreaming:
    async def test_final_tool_argument_fragment_with_usage_is_kept(self, mode: str) -> None:
        chunks = [
            make_chunk(tool_calls=[tool_delta(0, '{"city": ', call_id="call_1", name="get_weather")]),
            make_chunk(tool_calls=[tool_delta(0, '"Paris"}')], finish_reason="tool_calls", usage=USAGE),
        ]
        result, event, _, llm = await run_stream(mode, chunks)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "get_weather"
        assert json.loads(result[0]["function"]["arguments"]) == {"city": "Paris"}
        assert event["finish_reason"] == "tool_calls"
        assert_usage_counted_once(llm)

    async def test_multiple_tool_calls_finishing_in_usage_chunk(self, mode: str) -> None:
        chunks = [
            make_chunk(tool_calls=[tool_delta(0, '{"a": ', call_id="call_a", name="fa")]),
            make_chunk(tool_calls=[tool_delta(1, '{"b": ', call_id="call_b", name="fb")]),
            make_chunk(
                tool_calls=[tool_delta(0, "1}"), tool_delta(1, "2}")],
                finish_reason="tool_calls",
                usage=USAGE,
            ),
        ]
        result, _, _, _ = await run_stream(mode, chunks)
        args = {c["function"]["name"]: json.loads(c["function"]["arguments"]) for c in result}
        assert args == {"fa": {"a": 1}, "fb": {"b": 2}}

    async def test_tool_executes_with_complete_arguments(self, mode: str) -> None:
        seen: list[dict[str, Any]] = []

        def get_weather(city: str) -> str:
            seen.append({"city": city})
            return f"sunny in {city}"

        chunks = [
            make_chunk(tool_calls=[tool_delta(0, '{"ci', call_id="call_1", name="get_weather")]),
            make_chunk(tool_calls=[tool_delta(0, 'ty": "Paris"}')], finish_reason="tool_calls", usage=USAGE),
        ]
        result, _, _, _ = await run_stream(
            mode, chunks, available_functions={"get_weather": get_weather}
        )
        assert result == "sunny in Paris"
        assert seen == [{"city": "Paris"}]


class Answer(BaseModel):
    city: str
    temperature: int


class TestAsyncStructuredOutputStreaming:
    async def test_json_suffix_with_usage_is_kept(self) -> None:
        chunks = [
            make_chunk('{"city": "Paris", '),
            make_chunk('"temperature": 21}', finish_reason="stop", usage=USAGE),
        ]
        result, event, _, llm = await run_stream("async", chunks, response_model=Answer)
        assert result == Answer(city="Paris", temperature=21)
        assert event["finish_reason"] == "stop"
        assert_usage_counted_once(llm)

    async def test_usage_only_trailer_still_works(self) -> None:
        chunks = [
            make_chunk('{"city": "Paris", "temperature": 21}', finish_reason="stop"),
            make_chunk(usage=USAGE, no_choices=True),
        ]
        result, event, _, llm = await run_stream("async", chunks, response_model=Answer)
        assert result == Answer(city="Paris", temperature=21)
        assert event["finish_reason"] == "stop"
        assert_usage_counted_once(llm)
