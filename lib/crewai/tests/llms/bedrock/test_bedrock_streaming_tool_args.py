"""Tests for Bedrock streaming tool argument parsing (fixes #6149).

The Bedrock Converse streaming API delivers tool-call arguments as JSON
string fragments across ``contentBlockDelta`` events.  At
``contentBlockStop`` the handler must parse the accumulated string and
use the result — not read ``current_tool_use["input"]`` which is always
empty in the streaming path.
"""

import os
import json
import pytest
from unittest.mock import patch, MagicMock

from crewai.llm import LLM


@pytest.fixture(autouse=True)
def mock_aws_credentials():
    with patch.dict(os.environ, {
        "AWS_ACCESS_KEY_ID": "test-access-key",
        "AWS_SECRET_ACCESS_KEY": "test-secret-key",
        "AWS_DEFAULT_REGION": "us-east-1",
    }):
        with patch(
            "crewai.llms.providers.bedrock.completion.Session"
        ) as mock_session_class:
            mock_session_instance = MagicMock()
            mock_client = MagicMock()

            default_response = {
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [{"text": "Test response"}],
                    }
                },
                "usage": {
                    "inputTokens": 10,
                    "outputTokens": 5,
                    "totalTokens": 15,
                },
            }
            mock_client.converse.return_value = default_response
            mock_client.converse_stream.return_value = {"stream": []}
            mock_session_instance.client.return_value = mock_client
            mock_session_class.return_value = mock_session_instance
            yield mock_session_class, mock_client


def _build_stream_events(tool_name, tool_use_id, arg_chunks):
    """Build a synthetic Converse stream that delivers tool args in chunks.

    ``arg_chunks`` is a list of JSON string fragments that, when
    concatenated, form the complete arguments object.
    """
    events = [
        {"messageStart": {"role": "assistant"}},
        {
            "contentBlockStart": {
                "contentBlockIndex": 0,
                "start": {
                    "toolUse": {
                        "toolUseId": tool_use_id,
                        "name": tool_name,
                    }
                },
            }
        },
    ]
    for chunk in arg_chunks:
        events.append(
            {
                "contentBlockDelta": {
                    "delta": {"toolUse": {"input": chunk}},
                    "contentBlockIndex": 0,
                }
            }
        )
    events.append({"contentBlockStop": {"contentBlockIndex": 0}})
    events.append({"messageStop": {"stopReason": "tool_use"}})
    events.append(
        {
            "metadata": {
                "usage": {
                    "inputTokens": 100,
                    "outputTokens": 50,
                    "totalTokens": 150,
                }
            }
        }
    )
    return events


def test_streaming_tool_args_parsed_from_deltas():
    """Tool arguments accumulated across contentBlockDelta events must be
    passed to the tool function — not the empty dict from
    contentBlockStart.
    """
    llm = LLM(model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0", stream=True)

    received_args = {}

    def mock_weather_tool(city: str) -> str:
        received_args.update({"city": city})
        return f"Sunny in {city}"

    available_functions = {"get_weather": mock_weather_tool}

    arg_chunks = ['{"city":', ' "Paris"}']
    stream_events = _build_stream_events("get_weather", "tool-001", arg_chunks)

    final_response = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "It is sunny in Paris."}],
            }
        },
        "usage": {"inputTokens": 120, "outputTokens": 30, "totalTokens": 150},
    }

    with patch.object(llm._client, "converse_stream") as mock_stream, \
         patch.object(llm._client, "converse") as mock_converse:
        mock_stream.return_value = {"stream": iter(stream_events)}
        mock_converse.return_value = final_response

        messages = [{"role": "user", "content": "What's the weather in Paris?"}]
        result = llm.call(messages=messages, available_functions=available_functions)

    assert received_args == {"city": "Paris"}
    assert "sunny" in result.lower() or "paris" in result.lower()


def test_streaming_tool_args_single_chunk():
    """When the full JSON arrives in one delta, parsing still works."""
    llm = LLM(model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0", stream=True)

    received_args = {}

    def mock_search(query: str, limit: int) -> str:
        received_args.update({"query": query, "limit": limit})
        return "results"

    available_functions = {"search": mock_search}

    full_json = json.dumps({"query": "crewai bedrock", "limit": 10})
    stream_events = _build_stream_events("search", "tool-002", [full_json])

    final_response = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "Here are the results."}],
            }
        },
        "usage": {"inputTokens": 50, "outputTokens": 20, "totalTokens": 70},
    }

    with patch.object(llm._client, "converse_stream") as mock_stream, \
         patch.object(llm._client, "converse") as mock_converse:
        mock_stream.return_value = {"stream": iter(stream_events)}
        mock_converse.return_value = final_response

        messages = [{"role": "user", "content": "Search for crewai bedrock"}]
        result = llm.call(messages=messages, available_functions=available_functions)

    assert received_args == {"query": "crewai bedrock", "limit": 10}


def test_streaming_tool_args_many_chunks():
    """Arguments split across many small deltas are reassembled correctly."""
    llm = LLM(model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0", stream=True)

    received_args = {}

    def mock_tool(name: str, value: int) -> str:
        received_args.update({"name": name, "value": value})
        return "ok"

    available_functions = {"my_tool": mock_tool}

    arg_chunks = ["{", '"name"', ": ", '"hello', ' world"', ", ", '"value"', ": ", "42", "}"]
    stream_events = _build_stream_events("my_tool", "tool-003", arg_chunks)

    final_response = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "Done."}],
            }
        },
        "usage": {"inputTokens": 30, "outputTokens": 10, "totalTokens": 40},
    }

    with patch.object(llm._client, "converse_stream") as mock_stream, \
         patch.object(llm._client, "converse") as mock_converse:
        mock_stream.return_value = {"stream": iter(stream_events)}
        mock_converse.return_value = final_response

        messages = [{"role": "user", "content": "Do the thing"}]
        llm.call(messages=messages, available_functions=available_functions)

    assert received_args == {"name": "hello world", "value": 42}


def test_streaming_tool_use_input_set_on_current_tool_use():
    """After parsing, current_tool_use['input'] must be populated so the
    assistant message appended to the conversation carries the real args.
    """
    llm = LLM(model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0", stream=True)

    def mock_tool(x: int) -> str:
        return str(x * 2)

    available_functions = {"double": mock_tool}

    arg_chunks = ['{"x": 21}']
    stream_events = _build_stream_events("double", "tool-004", arg_chunks)

    final_response = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "42"}],
            }
        },
        "usage": {"inputTokens": 30, "outputTokens": 5, "totalTokens": 35},
    }

    with patch.object(llm._client, "converse_stream") as mock_stream, \
         patch.object(llm._client, "converse") as mock_converse:
        mock_stream.return_value = {"stream": iter(stream_events)}
        mock_converse.return_value = final_response

        messages = [{"role": "user", "content": "Double 21"}]
        llm.call(messages=messages, available_functions=available_functions)

    # The second call (non-streaming follow-up) should have received
    # messages containing the assistant's toolUse block with populated input.
    follow_up_messages = mock_converse.call_args[1].get(
        "messages", mock_converse.call_args[0][0] if mock_converse.call_args[0] else []
    )
    assistant_msg = next(
        (m for m in follow_up_messages if m["role"] == "assistant"), None
    )
    assert assistant_msg is not None
    tool_use_block = assistant_msg["content"][0]["toolUse"]
    assert tool_use_block["input"] == {"x": 21}
