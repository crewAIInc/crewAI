"""Tests for streaming tool call handling when available_functions is None.

Covers the fix for GitHub issue #4442: async streaming fails with tool/function calls
when available_functions is not provided (i.e., when the executor handles tool execution
instead of the LLM provider).

The fix ensures that streaming methods return accumulated tool calls in the correct
format instead of falling through and returning None/empty.
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crewai.llms.providers.openai.completion import OpenAICompletion


def _make_completion_chunk(
    tool_call_index: int = 0,
    tool_call_id: str | None = None,
    function_name: str | None = None,
    function_arguments: str | None = None,
    content: str | None = None,
    has_usage: bool = False,
) -> MagicMock:
    """Create a mock ChatCompletionChunk for streaming."""
    chunk = MagicMock()
    chunk.id = "chatcmpl-test123"

    if has_usage:
        chunk.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        chunk.choices = []
        return chunk

    chunk.usage = None

    choice = MagicMock()
    delta = MagicMock()

    delta.content = content

    if function_name is not None or function_arguments is not None or tool_call_id is not None:
        tc = MagicMock()
        tc.index = tool_call_index
        tc.id = tool_call_id
        tc.function = MagicMock()
        tc.function.name = function_name
        tc.function.arguments = function_arguments
        delta.tool_calls = [tc]
    else:
        delta.tool_calls = None

    choice.delta = delta
    chunk.choices = [choice]
    return chunk


def _make_responses_events(
    function_name: str = "get_temperature",
    function_args: str = '{"city": "Paris"}',
    call_id: str = "call_abc123",
) -> list[MagicMock]:
    """Create mock Responses API streaming events with a function call."""
    created_event = MagicMock()
    created_event.type = "response.created"
    created_event.response = MagicMock(id="resp_test123")

    args_delta_event = MagicMock()
    args_delta_event.type = "response.function_call_arguments.delta"

    item_done_event = MagicMock()
    item_done_event.type = "response.output_item.done"
    item_done_event.item = MagicMock()
    item_done_event.item.type = "function_call"
    item_done_event.item.call_id = call_id
    item_done_event.item.name = function_name
    item_done_event.item.arguments = function_args

    completed_event = MagicMock()
    completed_event.type = "response.completed"
    completed_event.response = MagicMock()
    completed_event.response.id = "resp_test123"
    completed_event.response.usage = MagicMock(
        input_tokens=10, output_tokens=5, total_tokens=15
    )

    return [created_event, args_delta_event, item_done_event, completed_event]


class TestStreamingCompletionToolCallsNoAvailableFunctions:
    """Tests for _handle_streaming_completion returning tool calls when available_functions is None."""

    def test_streaming_completion_returns_tool_calls_when_no_available_functions(self):
        """When streaming with tool calls and available_functions=None,
        the method should return formatted tool calls list."""
        llm = OpenAICompletion(model="gpt-4o")

        chunks = [
            _make_completion_chunk(
                tool_call_index=0,
                tool_call_id="call_abc123",
                function_name="get_temperature",
                function_arguments="",
            ),
            _make_completion_chunk(
                tool_call_index=0,
                function_arguments='{"city":',
            ),
            _make_completion_chunk(
                tool_call_index=0,
                function_arguments=' "Paris"}',
            ),
            _make_completion_chunk(has_usage=True),
        ]

        mock_stream = MagicMock()
        mock_stream.__iter__ = MagicMock(return_value=iter(chunks))

        with patch.object(
            llm.client.chat.completions, "create", return_value=mock_stream
        ):
            result = llm._handle_streaming_completion(
                params={"messages": [{"role": "user", "content": "test"}], "stream": True},
                available_functions=None,
            )

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_temperature"
        assert result[0]["function"]["arguments"] == '{"city": "Paris"}'
        assert result[0]["id"] == "call_abc123"

    def test_streaming_completion_multiple_tool_calls_no_available_functions(self):
        """When streaming with multiple tool calls and available_functions=None,
        all tool calls should be returned."""
        llm = OpenAICompletion(model="gpt-4o")

        chunks = [
            _make_completion_chunk(
                tool_call_index=0,
                tool_call_id="call_1",
                function_name="get_temperature",
                function_arguments='{"city": "Paris"}',
            ),
            _make_completion_chunk(
                tool_call_index=1,
                tool_call_id="call_2",
                function_name="get_temperature",
                function_arguments='{"city": "London"}',
            ),
            _make_completion_chunk(has_usage=True),
        ]

        mock_stream = MagicMock()
        mock_stream.__iter__ = MagicMock(return_value=iter(chunks))

        with patch.object(
            llm.client.chat.completions, "create", return_value=mock_stream
        ):
            result = llm._handle_streaming_completion(
                params={"messages": [{"role": "user", "content": "test"}], "stream": True},
                available_functions=None,
            )

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "get_temperature"
        assert result[0]["function"]["arguments"] == '{"city": "Paris"}'
        assert result[1]["function"]["name"] == "get_temperature"
        assert result[1]["function"]["arguments"] == '{"city": "London"}'

    def test_streaming_completion_with_available_functions_still_executes(self):
        """When available_functions IS provided, tool should be executed as before."""
        llm = OpenAICompletion(model="gpt-4o")

        chunks = [
            _make_completion_chunk(
                tool_call_index=0,
                tool_call_id="call_abc",
                function_name="get_temperature",
                function_arguments='{"city": "Paris"}',
            ),
            _make_completion_chunk(has_usage=True),
        ]

        mock_stream = MagicMock()
        mock_stream.__iter__ = MagicMock(return_value=iter(chunks))

        with patch.object(
            llm.client.chat.completions, "create", return_value=mock_stream
        ), patch.object(
            llm, "_handle_tool_execution", return_value="72F in Paris"
        ) as mock_exec:
            result = llm._handle_streaming_completion(
                params={"messages": [{"role": "user", "content": "test"}], "stream": True},
                available_functions={"get_temperature": lambda city: f"72F in {city}"},
            )

        assert result == "72F in Paris"
        mock_exec.assert_called_once()


class TestAsyncStreamingCompletionToolCallsNoAvailableFunctions:
    """Tests for _ahandle_streaming_completion returning tool calls when available_functions is None."""

    @pytest.mark.asyncio
    async def test_async_streaming_completion_returns_tool_calls_when_no_available_functions(self):
        """When async streaming with tool calls and available_functions=None,
        the method should return formatted tool calls list."""
        llm = OpenAICompletion(model="gpt-4o")

        chunks = [
            _make_completion_chunk(
                tool_call_index=0,
                tool_call_id="call_abc123",
                function_name="get_temperature",
                function_arguments="",
            ),
            _make_completion_chunk(
                tool_call_index=0,
                function_arguments='{"city":',
            ),
            _make_completion_chunk(
                tool_call_index=0,
                function_arguments=' "Paris"}',
            ),
            _make_completion_chunk(has_usage=True),
        ]

        async def mock_aiter():
            for c in chunks:
                yield c

        with patch.object(
            llm.async_client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_aiter(),
        ):
            result = await llm._ahandle_streaming_completion(
                params={"messages": [{"role": "user", "content": "test"}], "stream": True},
                available_functions=None,
            )

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_temperature"
        assert result[0]["function"]["arguments"] == '{"city": "Paris"}'
        assert result[0]["id"] == "call_abc123"

    @pytest.mark.asyncio
    async def test_async_streaming_completion_multiple_tool_calls_no_available_functions(self):
        """When async streaming with multiple tool calls and available_functions=None,
        all tool calls should be returned."""
        llm = OpenAICompletion(model="gpt-4o")

        chunks = [
            _make_completion_chunk(
                tool_call_index=0,
                tool_call_id="call_1",
                function_name="get_temperature",
                function_arguments='{"city": "Paris"}',
            ),
            _make_completion_chunk(
                tool_call_index=1,
                tool_call_id="call_2",
                function_name="get_temperature",
                function_arguments='{"city": "London"}',
            ),
            _make_completion_chunk(has_usage=True),
        ]

        async def mock_aiter():
            for c in chunks:
                yield c

        with patch.object(
            llm.async_client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_aiter(),
        ):
            result = await llm._ahandle_streaming_completion(
                params={"messages": [{"role": "user", "content": "test"}], "stream": True},
                available_functions=None,
            )

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "get_temperature"
        assert result[1]["function"]["name"] == "get_temperature"
        assert result[0]["function"]["arguments"] == '{"city": "Paris"}'
        assert result[1]["function"]["arguments"] == '{"city": "London"}'


class TestStreamingResponsesToolCallsNoAvailableFunctions:
    """Tests for _handle_streaming_responses returning function calls when available_functions is None."""

    def test_streaming_responses_returns_function_calls_when_no_available_functions(self):
        """When streaming Responses API with function calls and available_functions=None,
        the method should return function_calls list."""
        llm = OpenAICompletion(model="gpt-4o", use_responses_api=True)

        events = _make_responses_events(
            function_name="get_temperature",
            function_args='{"city": "Paris"}',
            call_id="call_abc123",
        )

        mock_stream = MagicMock()
        mock_stream.__iter__ = MagicMock(return_value=iter(events))

        with patch.object(
            llm.client.responses, "create", return_value=mock_stream
        ):
            result = llm._handle_streaming_responses(
                params={"input": [{"role": "user", "content": "test"}], "model": "gpt-4o", "stream": True},
                available_functions=None,
            )

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["name"] == "get_temperature"
        assert result[0]["arguments"] == '{"city": "Paris"}'
        assert result[0]["id"] == "call_abc123"

    def test_streaming_responses_with_available_functions_still_executes(self):
        """When available_functions IS provided, tool should be executed as before."""
        llm = OpenAICompletion(model="gpt-4o", use_responses_api=True)

        events = _make_responses_events(
            function_name="get_temperature",
            function_args='{"city": "Paris"}',
        )

        mock_stream = MagicMock()
        mock_stream.__iter__ = MagicMock(return_value=iter(events))

        with patch.object(
            llm.client.responses, "create", return_value=mock_stream
        ), patch.object(
            llm, "_handle_tool_execution", return_value="72F in Paris"
        ) as mock_exec:
            result = llm._handle_streaming_responses(
                params={"input": [{"role": "user", "content": "test"}], "model": "gpt-4o", "stream": True},
                available_functions={"get_temperature": lambda city: f"72F in {city}"},
            )

        assert result == "72F in Paris"
        mock_exec.assert_called_once()


class TestAsyncStreamingResponsesToolCallsNoAvailableFunctions:
    """Tests for _ahandle_streaming_responses returning function calls when available_functions is None."""

    @pytest.mark.asyncio
    async def test_async_streaming_responses_returns_function_calls_when_no_available_functions(self):
        """When async streaming Responses API with function calls and available_functions=None,
        the method should return function_calls list."""
        llm = OpenAICompletion(model="gpt-4o", use_responses_api=True)

        events = _make_responses_events(
            function_name="get_temperature",
            function_args='{"city": "Paris"}',
            call_id="call_abc123",
        )

        async def mock_aiter():
            for e in events:
                yield e

        with patch.object(
            llm.async_client.responses,
            "create",
            new_callable=AsyncMock,
            return_value=mock_aiter(),
        ):
            result = await llm._ahandle_streaming_responses(
                params={"input": [{"role": "user", "content": "test"}], "model": "gpt-4o", "stream": True},
                available_functions=None,
            )

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["name"] == "get_temperature"
        assert result[0]["arguments"] == '{"city": "Paris"}'
        assert result[0]["id"] == "call_abc123"

    @pytest.mark.asyncio
    async def test_async_streaming_responses_multiple_function_calls_no_available_functions(self):
        """When async streaming Responses API with multiple function calls and available_functions=None,
        all function calls should be returned."""
        llm = OpenAICompletion(model="gpt-4o", use_responses_api=True)

        created_event = MagicMock()
        created_event.type = "response.created"
        created_event.response = MagicMock(id="resp_test")

        item1 = MagicMock()
        item1.type = "response.output_item.done"
        item1.item = MagicMock()
        item1.item.type = "function_call"
        item1.item.call_id = "call_1"
        item1.item.name = "get_temperature"
        item1.item.arguments = '{"city": "Paris"}'

        item2 = MagicMock()
        item2.type = "response.output_item.done"
        item2.item = MagicMock()
        item2.item.type = "function_call"
        item2.item.call_id = "call_2"
        item2.item.name = "get_temperature"
        item2.item.arguments = '{"city": "London"}'

        completed = MagicMock()
        completed.type = "response.completed"
        completed.response = MagicMock()
        completed.response.id = "resp_test"
        completed.response.usage = MagicMock(
            input_tokens=10, output_tokens=5, total_tokens=15
        )

        events = [created_event, item1, item2, completed]

        async def mock_aiter():
            for e in events:
                yield e

        with patch.object(
            llm.async_client.responses,
            "create",
            new_callable=AsyncMock,
            return_value=mock_aiter(),
        ):
            result = await llm._ahandle_streaming_responses(
                params={"input": [{"role": "user", "content": "test"}], "model": "gpt-4o", "stream": True},
                available_functions=None,
            )

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["name"] == "get_temperature"
        assert result[0]["arguments"] == '{"city": "Paris"}'
        assert result[1]["name"] == "get_temperature"
        assert result[1]["arguments"] == '{"city": "London"}'
