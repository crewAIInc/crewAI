"""Regression tests for parallel tool calls in Azure streaming completions.

Azure AI Inference is OpenAI-compatible: each streamed chunk carries a
``tool_call`` whose own ``index`` field identifies the call the delta belongs
to. The streaming accumulator must key on that wire index, not on the position
of the tool_call inside the chunk, otherwise parallel tool calls collapse into
a single corrupted call.
"""

import json
from unittest.mock import patch

from azure.ai.inference.models import StreamingChatCompletionsUpdate

from crewai.llms.providers.azure.completion import AzureCompletion


def _tool_call_chunk(index: int, call_id: str, name: str, arguments: str):
    """Build one streaming update carrying a single tool-call delta."""
    return StreamingChatCompletionsUpdate(
        id="chatcmpl-1",
        model="gpt-4o",
        created=None,
        choices=[
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "index": index,
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        }
                    ],
                },
                "finish_reason": None,
            }
        ],
    )


def test_azure_streaming_parallel_tool_calls_are_keyed_by_wire_index():
    """Both parallel tool calls must reach the executor intact.

    Each chunk carries exactly one tool_call tagged with its own wire ``index``
    (0 and 1). Aggregating by position inside the chunk puts every delta in
    slot 0, so the executor receives a single call holding the first call's id,
    the last call's name, and the two argument fragments concatenated into
    invalid JSON.
    """
    chunks = [
        _tool_call_chunk(0, "call_aaa", "get_weather", '{"city":'),
        _tool_call_chunk(0, "call_aaa", "get_weather", '"Paris"}'),
        _tool_call_chunk(1, "call_bbb", "get_time", '{"tz":'),
        _tool_call_chunk(1, "call_bbb", "get_time", '"UTC"}'),
    ]

    llm = AzureCompletion(
        model="gpt-4o",
        api_key="test-key",
        endpoint="https://test.openai.azure.com",
        stream=True,
    )

    with patch.object(llm._client, "complete") as mock_complete:
        mock_complete.return_value = iter(chunks)

        result = llm.call([{"role": "user", "content": "Weather and time in Paris?"}])

    assert isinstance(result, list)
    assert len(result) == 2
    assert [call["id"] for call in result] == ["call_aaa", "call_bbb"]

    weather, clock = result
    assert weather["function"]["name"] == "get_weather"
    assert json.loads(weather["function"]["arguments"]) == {"city": "Paris"}
    assert clock["function"]["name"] == "get_time"
    assert json.loads(clock["function"]["arguments"]) == {"tz": "UTC"}
