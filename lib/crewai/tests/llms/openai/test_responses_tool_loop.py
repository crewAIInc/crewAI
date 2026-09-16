"""Tests for tool calling on the OpenAI Responses API path.

The two APIs express tool calling differently:

    Chat Completions   assistant message with `tool_calls` (content: None),
                       then {"role": "tool", "tool_call_id": ...}
    Responses          flat {"type": "function_call", "call_id", ...} and
                       {"type": "function_call_output", "call_id", "output"} items

Sending the chat shape to /v1/responses is rejected outright:

    Invalid type for 'input[1].content': expected one of an array of objects or
    string, but got null instead.

Verified against the live endpoint: the chat shape 400s, the native items complete.
"""

import asyncio
from types import SimpleNamespace

import pytest

from crewai.llms.providers.openai.completion import OpenAICompletion
from crewai.utilities.agent_utils import extract_tool_call_info, is_tool_call_list


# The shape OpenAICompletion._extract_function_calls_from_response builds from a
# Responses payload: no nested "function" object, no "input".
RESPONSES_TOOL_CALL = {
    "id": "call_abc",
    "name": "multiply",
    "arguments": '{"a": 17, "b": 23}',
}

# A raw Responses `function_call` output item, as returned by the API. Note that
# "id" and "call_id" are different values -- the matching function_call_output must
# reference "call_id".
RAW_RESPONSES_ITEM = {
    "type": "function_call",
    "id": "fc_0adeb715c5d740c7006a65ccb7",
    "call_id": "call_dEoHFrYnOgWYvk17FymdcDZ5",
    "name": "multiply",
    "arguments": '{"a": 17, "b": 23}',
    "status": "completed",
}


def build(model: str = "gpt-5.5", **kwargs) -> OpenAICompletion:
    return OpenAICompletion(model=model, api_key="sk-test", api="responses", **kwargs)


class TestToolCallRecognition:
    """The executor must recognize Responses-shaped tool calls."""

    def test_recognizes_responses_shape(self):
        assert is_tool_call_list([RESPONSES_TOOL_CALL])

    def test_extracts_arguments_from_top_level(self):
        """Previously fell through to `input` and silently yielded {}."""
        call_id, name, args = extract_tool_call_info(RESPONSES_TOOL_CALL)

        assert call_id == "call_abc"
        assert name == "multiply"
        assert args == '{"a": 17, "b": 23}'

    @pytest.mark.parametrize(
        ("tool_call", "expected_args"),
        [
            (
                {"id": "c", "function": {"name": "f", "arguments": '{"x":1}'}},
                '{"x":1}',
            ),
            ({"toolUseId": "c", "name": "f", "input": {"x": 1}}, {"x": 1}),
        ],
    )
    def test_other_provider_shapes_still_work(self, tool_call, expected_args):
        """Chat Completions and Bedrock shapes must be unaffected."""
        assert is_tool_call_list([tool_call])
        assert extract_tool_call_info(tool_call)[2] == expected_args

    def test_raw_responses_item_uses_call_id_not_item_id(self):
        """A raw function_call item carries both; only call_id can be correlated.

        function_call_output must reference call_id, so picking up the item's own
        "id" (fc_...) would produce a tool result the model can't match to its
        invocation.
        """
        call_id, name, args = extract_tool_call_info(RAW_RESPONSES_ITEM)

        assert call_id == "call_dEoHFrYnOgWYvk17FymdcDZ5"
        assert call_id != RAW_RESPONSES_ITEM["id"]
        assert name == "multiply"
        assert args == '{"a": 17, "b": 23}'


class TestResponsesInputTranslation:
    """Chat-format tool messages must become native Responses items."""

    def test_assistant_tool_calls_become_function_call_items(self):
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "multiply", "arguments": '{"a":17,"b":23}'},
                }
            ],
        }

        assert OpenAICompletion._to_responses_input(message) == [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "multiply",
                "arguments": '{"a":17,"b":23}',
            }
        ]

    def test_tool_result_becomes_function_call_output(self):
        message = {"role": "tool", "tool_call_id": "call_1", "content": "391"}

        assert OpenAICompletion._to_responses_input(message) == [
            {"type": "function_call_output", "call_id": "call_1", "output": "391"}
        ]

    def test_assistant_text_alongside_tool_calls_is_preserved(self):
        message = {
            "role": "assistant",
            "content": "Let me calculate.",
            "tool_calls": [
                {"id": "c1", "function": {"name": "multiply", "arguments": "{}"}}
            ],
        }

        items = OpenAICompletion._to_responses_input(message)

        assert items[0] == {"role": "assistant", "content": "Let me calculate."}
        assert items[1]["type"] == "function_call"

    def test_parallel_tool_calls_become_separate_items(self):
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "function": {"name": "multiply", "arguments": "{}"}},
                {"id": "c2", "function": {"name": "add", "arguments": "{}"}},
            ],
        }

        items = OpenAICompletion._to_responses_input(message)

        assert [i["call_id"] for i in items] == ["c1", "c2"]

    @pytest.mark.parametrize(
        "message",
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
    )
    def test_messages_without_tool_calls_pass_through(self, message):
        assert OpenAICompletion._to_responses_input(message) == [message]

    def test_non_string_tool_output_is_coerced(self):
        """Tool results arrive as ints, dicts, etc. The API requires a string."""
        message = {"role": "tool", "tool_call_id": "c1", "content": 391}

        assert OpenAICompletion._to_responses_input(message)[0]["output"] == "391"

    def test_call_and_output_ids_round_trip(self):
        """The id extracted from a call must be the one sent back with its result.

        This is the correlation the API relies on: a function_call_output whose
        call_id doesn't match an emitted function_call is rejected or ignored.
        """
        call_id, name, args = extract_tool_call_info(RAW_RESPONSES_ITEM)

        assistant = OpenAICompletion._to_responses_input(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": call_id, "function": {"name": name, "arguments": args}}
                ],
            }
        )
        result = OpenAICompletion._to_responses_input(
            {"role": "tool", "tool_call_id": call_id, "content": "391"}
        )

        assert assistant[0]["call_id"] == result[0]["call_id"] == call_id


class TestPreparedParams:
    """End-to-end shape of the `input` list handed to the Responses API."""

    def test_tool_conversation_produces_valid_input(self):
        llm = build()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "multiply 17 and 23"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "multiply", "arguments": '{"a":17,"b":23}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "391"},
        ]

        params = llm._prepare_responses_params(messages)

        assert params["instructions"] == "You are helpful."
        assert [item.get("type") or item["role"] for item in params["input"]] == [
            "user",
            "function_call",
            "function_call_output",
        ]
        # The rejected shape was an item carrying an explicit content: None.
        # function_call items have no content key at all, which is valid.
        assert not any(
            "content" in item and item["content"] is None for item in params["input"]
        )


class TestResponsesStreamingToolCalls:
    """Responses streaming must return function calls when available_functions is None."""

    @pytest.fixture
    def mock_responses_stream(self):
        fc_item = SimpleNamespace(
            type="function_call",
            id="fc_1",
            call_id="call_abc",
            name="multiply",
            arguments='{"a": 17, "b": 23}',
            status="completed",
        )
        response = SimpleNamespace(
            id="resp_1",
            status="completed",
            output=[fc_item],
            output_text="",
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
                input_tokens_details=None,
                output_tokens_details=None,
            ),
        )

        def sync_events():
            yield SimpleNamespace(type="response.created", response=response)
            yield SimpleNamespace(type="response.output_item.done", item=fc_item)
            yield SimpleNamespace(type="response.completed", response=response)

        async def async_events():
            yield SimpleNamespace(type="response.created", response=response)
            yield SimpleNamespace(type="response.output_item.done", item=fc_item)
            yield SimpleNamespace(type="response.completed", response=response)

        return response, sync_events, async_events

    def test_sync_streaming_returns_tool_calls_when_available_functions_none(
        self, mock_responses_stream
    ):
        _, sync_events, _ = mock_responses_stream

        class FakeResponses:
            def create(self, **kwargs):
                return sync_events()

        class FakeClient:
            def __init__(self):
                self.responses = FakeResponses()

        llm = build(stream=True)
        llm._get_sync_client = lambda: FakeClient()

        result = llm._handle_streaming_responses(
            params={"input": [{"role": "user", "content": "multiply"}]},
            available_functions=None,
        )

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["name"] == "multiply"
        assert result[0]["arguments"] == '{"a": 17, "b": 23}'
        assert result[0]["id"] == "call_abc"

    def test_sync_streaming_executes_function_when_available_functions_provided(
        self, mock_responses_stream
    ):
        _, sync_events, _ = mock_responses_stream

        class FakeResponses:
            def create(self, **kwargs):
                return sync_events()

        class FakeClient:
            def __init__(self):
                self.responses = FakeResponses()

        llm = build(stream=True)
        llm._get_sync_client = lambda: FakeClient()

        executed_args = []

        def fake_multiply(a, b):
            executed_args.append((a, b))
            return 391

        result = llm._handle_streaming_responses(
            params={"input": [{"role": "user", "content": "multiply"}]},
            available_functions={"multiply": fake_multiply},
        )

        assert result == "391"
        assert executed_args == [(17, 23)]

    def test_async_streaming_returns_tool_calls_when_available_functions_none(
        self, mock_responses_stream
    ):
        _, _, async_events = mock_responses_stream

        class FakeAsyncResponses:
            async def create(self, **kwargs):
                return async_events()

        class FakeAsyncClient:
            def __init__(self):
                self.responses = FakeAsyncResponses()

        llm = build(stream=True)
        llm._get_async_client = lambda: FakeAsyncClient()

        async def run_test():
            return await llm._ahandle_streaming_responses(
                params={"input": [{"role": "user", "content": "multiply"}]},
                available_functions=None,
            )

        result = asyncio.run(run_test())

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["name"] == "multiply"
        assert result[0]["arguments"] == '{"a": 17, "b": 23}'
        assert result[0]["id"] == "call_abc"

    def test_async_streaming_executes_function_when_available_functions_provided(
        self, mock_responses_stream
    ):
        _, _, async_events = mock_responses_stream

        class FakeAsyncResponses:
            async def create(self, **kwargs):
                return async_events()

        class FakeAsyncClient:
            def __init__(self):
                self.responses = FakeAsyncResponses()

        llm = build(stream=True)
        llm._get_async_client = lambda: FakeAsyncClient()

        executed_args = []

        def fake_multiply(a, b):
            executed_args.append((a, b))
            return 391

        async def run_test():
            return await llm._ahandle_streaming_responses(
                params={"input": [{"role": "user", "content": "multiply"}]},
                available_functions={"multiply": fake_multiply},
            )

        result = asyncio.run(run_test())

        assert result == "391"
        assert executed_args == [(17, 23)]

