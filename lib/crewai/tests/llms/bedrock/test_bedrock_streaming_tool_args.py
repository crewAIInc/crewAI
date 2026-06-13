"""Regression tests for streaming Bedrock tool-call argument handling.

The streaming Converse handlers deliver tool input as a sequence of JSON
string deltas (``contentBlockDelta`` -> ``toolUse.input``) that are
accumulated separately from the tool-use block. These tests assert that the
accumulated input is folded back into the tool call at ``contentBlockStop``,
so executed tools receive their real arguments instead of an empty ``{}``.

This is the streaming counterpart of the non-streaming fix in #5415
(issue #4972).
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from crewai.llm import LLM
from crewai.llms.providers.bedrock.completion import BedrockCompletion


def _make_tool_use_stream() -> list[dict]:
    """Synthetic Converse stream: a single tool call with JSON-chunked input."""
    # Tool input is delivered as two partial JSON string fragments that only
    # form valid JSON once concatenated: '{"city":' + ' "Paris"}'.
    chunk1 = '{"city":'
    chunk2 = ' "Paris"}'
    return [
        {"messageStart": {"role": "assistant"}},
        {
            "contentBlockStart": {
                "start": {"toolUse": {"toolUseId": "tool-1", "name": "get_weather"}},
                "contentBlockIndex": 0,
            }
        },
        {"contentBlockDelta": {"delta": {"toolUse": {"input": chunk1}}}},
        {"contentBlockDelta": {"delta": {"toolUse": {"input": chunk2}}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "tool_use"}},
    ]


def _build_completion() -> BedrockCompletion:
    """Build a BedrockCompletion with mocked AWS credentials/session."""
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


def test_streaming_tool_call_preserves_arguments():
    """Sync streaming: function_args must carry the streamed tool input."""
    llm = _build_completion()

    captured: dict = {}

    def capture(function_args, **kwargs):
        captured["args"] = function_args
        return None  # returning None stops the recursive _handle_converse call

    mock_client = MagicMock()
    mock_client.converse_stream.return_value = {"stream": _make_tool_use_stream()}

    with (
        patch.object(llm, "_get_sync_client", return_value=mock_client),
        patch.object(llm, "_handle_tool_execution", side_effect=capture),
    ):
        llm._handle_streaming_converse(
            messages=[{"role": "user", "content": "weather in Paris?"}],
            body={},
            available_functions={"get_weather": lambda **kw: "sunny"},
        )

    assert captured["args"] == {"city": "Paris"}


@pytest.mark.asyncio
async def test_async_streaming_tool_call_preserves_arguments():
    """Async streaming: function_args must carry the streamed tool input."""
    llm = _build_completion()

    class _AsyncStream:
        def __init__(self, events):
            self._events = events

        def __aiter__(self):
            self._it = iter(self._events)
            return self

        async def __anext__(self):
            try:
                return next(self._it)
            except StopIteration:
                raise StopAsyncIteration

    async def _converse_stream(**kwargs):
        return {"stream": _AsyncStream(_make_tool_use_stream())}

    mock_async_client = MagicMock()
    mock_async_client.converse_stream = _converse_stream

    async def _ensure(*args, **kwargs):
        return mock_async_client

    captured: dict = {}

    def capture(function_args, **kwargs):
        captured["args"] = function_args
        return None

    with (
        patch.object(llm, "_ensure_async_client", side_effect=_ensure),
        patch.object(llm, "_handle_tool_execution", side_effect=capture),
    ):
        await llm._ahandle_streaming_converse(
            messages=[{"role": "user", "content": "weather in Paris?"}],
            body={},
            available_functions={"get_weather": lambda **kw: "sunny"},
        )

    assert captured["args"] == {"city": "Paris"}
