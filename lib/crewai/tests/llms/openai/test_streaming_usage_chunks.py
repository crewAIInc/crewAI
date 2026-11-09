"""Streaming chunks that carry usage must not drop their choices.

Chat Completions providers (e.g. DeepSeek) send the token usage on the same
chunk as the last text delta, tool-call delta or ``finish_reason``. The OpenAI
streaming handlers used to ``continue`` as soon as usage was seen, so whatever
else the chunk carried was silently lost.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from openai.types.chat import ChatCompletionChunk
from pydantic import BaseModel

from crewai.llms.providers.openai.completion import OpenAICompletion


MESSAGES = [{"role": "user", "content": "hi"}]
USAGE = {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}


class Answer(BaseModel):
    value: str


def chunk(
    text: str | None = None,
    *,
    with_usage: bool = False,
    finish_reason: str | None = None,
    tool_call: dict[str, Any] | None = None,
) -> ChatCompletionChunk:
    delta: dict[str, Any] = {"content": text}
    if tool_call:
        delta["tool_calls"] = [tool_call]
    choices = [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
    return ChatCompletionChunk.model_validate(
        {
            "id": "chunk-id",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "gpt-4o",
            "choices": choices,
            "usage": USAGE if with_usage else None,
        }
    )


def usage_only_chunk() -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate(
        {
            "id": "chunk-id",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "gpt-4o",
            "choices": [],
            "usage": USAGE,
        }
    )


def build_llm(stream: bool = True) -> OpenAICompletion:
    return OpenAICompletion(model="gpt-4o", api_key="sk-test", stream=stream)


def sync_client(chunks: list[ChatCompletionChunk]) -> SimpleNamespace:
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_: iter(chunks))
        )
    )


def async_client(chunks: list[ChatCompletionChunk]) -> SimpleNamespace:
    async def _aiter():
        for item in chunks:
            yield item

    async def _create(**_: Any):
        return _aiter()

    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_create))
    )


def run_sync(chunks: list[ChatCompletionChunk]):
    llm = build_llm()
    with (
        patch.object(llm, "_get_sync_client", return_value=sync_client(chunks)),
        patch.object(llm, "_emit_stream_chunk_event"),
        patch.object(llm, "_track_token_usage_internal") as tracked,
        patch.object(llm, "_emit_call_completed_event") as completed,
    ):
        result = llm._handle_streaming_completion({"messages": MESSAGES})
    return result, tracked, completed


@pytest.mark.asyncio
async def run_async(chunks: list[ChatCompletionChunk], response_model=None):
    llm = build_llm()
    with (
        patch.object(llm, "_get_async_client", return_value=async_client(chunks)),
        patch.object(llm, "_emit_stream_chunk_event"),
        patch.object(llm, "_track_token_usage_internal") as tracked,
        patch.object(llm, "_emit_call_completed_event") as completed,
    ):
        result = await llm._ahandle_streaming_completion(
            {"messages": MESSAGES}, response_model=response_model
        )
    return result, tracked, completed


def test_sync_stream_keeps_text_in_final_usage_chunk():
    result, tracked, completed = run_sync(
        [chunk("HELLO-"), chunk("END", with_usage=True, finish_reason="stop")]
    )

    assert result == "HELLO-END"
    assert completed.call_args.kwargs["finish_reason"] == "stop"
    tracked.assert_called_once_with(USAGE)


def test_sync_stream_keeps_single_text_chunk_with_usage():
    result, _, completed = run_sync(
        [chunk("HELLO-END", with_usage=True, finish_reason="stop")]
    )

    assert result == "HELLO-END"
    assert completed.call_args.kwargs["finish_reason"] == "stop"


def test_sync_stream_keeps_finish_reason_with_empty_delta_usage_chunk():
    result, _, completed = run_sync(
        [chunk("HELLO-END"), chunk("", with_usage=True, finish_reason="stop")]
    )

    assert result == "HELLO-END"
    assert completed.call_args.kwargs["finish_reason"] == "stop"


def test_sync_stream_keeps_tool_arguments_in_usage_chunk():
    result, tracked, completed = run_sync(
        [
            chunk(
                tool_call={
                    "index": 0,
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"city": "Par'},
                }
            ),
            chunk(
                with_usage=True,
                finish_reason="tool_calls",
                tool_call={
                    "index": 0,
                    "type": "function",
                    "function": {"arguments": 'is"}'},
                },
            ),
        ]
    )

    assert result == [
        {
            "id": "call-1",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"city": "Paris"}'},
            "index": 0,
        }
    ]
    assert completed.call_args.kwargs["finish_reason"] == "tool_calls"
    tracked.assert_called_once_with(USAGE)


def test_sync_stream_records_usage_only_chunk():
    result, tracked, _ = run_sync([chunk("HELLO-END"), usage_only_chunk()])

    assert result == "HELLO-END"
    tracked.assert_called_once_with(USAGE)


@pytest.mark.asyncio
async def test_async_stream_keeps_text_and_finish_reason_from_usage_chunk():
    result, tracked, completed = await run_async(
        [chunk("HELLO-"), chunk("END", with_usage=True, finish_reason="stop")]
    )

    assert result == "HELLO-END"
    assert completed.call_args.kwargs["finish_reason"] == "stop"
    tracked.assert_called_once_with(USAGE)


@pytest.mark.asyncio
async def test_async_stream_records_usage_only_chunk():
    result, tracked, _ = await run_async([chunk("HELLO-END"), usage_only_chunk()])

    assert result == "HELLO-END"
    tracked.assert_called_once_with(USAGE)


@pytest.mark.asyncio
async def test_async_response_model_keeps_json_suffix_from_usage_chunk():
    result, tracked, completed = await run_async(
        [
            chunk('{"value": "AB'),
            chunk('CD"}', with_usage=True, finish_reason="stop"),
        ],
        response_model=Answer,
    )

    assert result == Answer(value="ABCD")
    assert completed.call_args.kwargs["finish_reason"] == "stop"
    tracked.assert_called_once_with(USAGE)
