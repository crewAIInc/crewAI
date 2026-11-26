"""Behavioral tests for OpenAI-style image_url conversion in AnthropicCompletion."""

from __future__ import annotations

import copy
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from crewai.llms.providers.anthropic.completion import AnthropicCompletion
from crewai.tools.agent_tools.add_image_tool import AddImageTool
from crewai.utilities.types import LLMMessage


def _mock_text_response(text: str = "ok") -> MagicMock:
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = text

    response = MagicMock()
    response.content = [text_block]
    response.usage = MagicMock(
        input_tokens=10,
        output_tokens=5,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    response.stop_reason = "end_turn"
    response.id = "msg_test"
    return response


def _build_llm() -> tuple[AnthropicCompletion, MagicMock, MagicMock]:
    llm = AnthropicCompletion(model="claude-sonnet-4-6", api_key="test-key")
    sync_client = MagicMock()
    sync_client.messages.create.return_value = _mock_text_response()
    sync_client.messages.count_tokens.return_value = MagicMock(input_tokens=42)
    async_client = MagicMock()
    async_client.messages.create = AsyncMock(return_value=_mock_text_response())
    llm._client = sync_client
    llm._async_client = async_client
    return llm, sync_client, async_client


def test_remote_https_image_url_in_user_message_converts_to_anthropic_url_source() -> (
    None
):
    llm, sync_client, _ = _build_llm()
    messages: list[LLMMessage] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this chart"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/Chart.PNG?Token=AbC",
                        "detail": "high",
                    },
                },
            ],
        }
    ]
    original = copy.deepcopy(messages)

    assert llm.call(messages) == "ok"

    sent_messages = sync_client.messages.create.call_args.kwargs["messages"]
    assert sent_messages == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this chart"},
                {
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": "https://example.com/Chart.PNG?Token=AbC",
                    },
                },
            ],
        }
    ]
    assert messages == original


@pytest.mark.asyncio
async def test_remote_http_and_string_shorthand_image_url_in_acall() -> None:
    llm, _, async_client = _build_llm()
    messages: list[LLMMessage] = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "http://example.com/a.jpg"}},
                {"type": "image_url", "image_url": "HTTPS://example.com/b.png"},
            ],
        }
    ]

    assert await llm.acall(messages) == "ok"

    sent_messages = async_client.messages.create.call_args.kwargs["messages"]
    assert sent_messages == [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "url", "url": "http://example.com/a.jpg"},
                },
                {
                    "type": "image",
                    "source": {"type": "url", "url": "HTTPS://example.com/b.png"},
                },
            ],
        }
    ]


def test_remote_image_url_in_tool_result_converts_to_anthropic_url_source() -> None:
    llm, sync_client, _ = _build_llm()
    messages: list[LLMMessage] = [
        {"role": "user", "content": "Fetch the screenshot"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "toolu_1",
                    "type": "function",
                    "function": {"name": "screenshot", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "toolu_1",
            "content": [
                {"type": "text", "text": "Captured screenshot"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/screen.png"},
                },
            ],
        },
    ]

    assert llm.call(messages) == "ok"

    sent_messages = sync_client.messages.create.call_args.kwargs["messages"]
    tool_result = sent_messages[2]["content"][0]
    assert tool_result == {
        "type": "tool_result",
        "tool_use_id": "toolu_1",
        "content": [
            {"type": "text", "text": "Captured screenshot"},
            {
                "type": "image",
                "source": {"type": "url", "url": "https://example.com/screen.png"},
            },
        ],
    }


def test_add_image_tool_url_message_converts_to_anthropic_url_source() -> None:
    llm, sync_client, _ = _build_llm()
    tool_message: Any = AddImageTool()._run(
        image_url="https://example.com/diagram.png",
        action="Explain this diagram",
    )

    assert llm.call([tool_message]) == "ok"

    sent_messages = sync_client.messages.create.call_args.kwargs["messages"]
    assert sent_messages == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Explain this diagram"},
                {
                    "type": "image",
                    "source": {"type": "url", "url": "https://example.com/diagram.png"},
                },
            ],
        }
    ]


def test_data_uri_image_url_normalizes_media_type_parameters_case_and_default() -> None:
    llm, sync_client, _ = _build_llm()
    messages: list[LLMMessage] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ"},
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;charset=utf-8;base64,iVBORw=="
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "data:;base64,iVBORw=="},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "DATA:IMAGE/WEBP;BASE64,UklGRaB"},
                },
            ],
        }
    ]

    assert llm.call(messages) == "ok"

    sent_content = sync_client.messages.create.call_args.kwargs["messages"][0][
        "content"
    ]
    assert sent_content == [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": "/9j/4AAQ",
            },
        },
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "iVBORw==",
            },
        },
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "iVBORw==",
            },
        },
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/webp",
                "data": "UklGRaB",
            },
        },
    ]


def test_cache_control_preserved_on_converted_image_blocks() -> None:
    llm, sync_client, _ = _build_llm()
    messages: list[LLMMessage] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/cached.png"},
                    "cache_control": {"type": "ephemeral"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,iVBORw=="},
                    "cache_control": {"type": "ephemeral"},
                },
            ],
        }
    ]

    assert llm.call(messages) == "ok"

    sent_content = sync_client.messages.create.call_args.kwargs["messages"][0][
        "content"
    ]
    assert sent_content == [
        {
            "type": "image",
            "source": {"type": "url", "url": "https://example.com/cached.png"},
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "iVBORw==",
            },
            "cache_control": {"type": "ephemeral"},
        },
    ]


def test_native_and_malformed_image_blocks_pass_through_without_crashing() -> None:
    llm, sync_client, _ = _build_llm()
    native_block = {
        "type": "image",
        "source": {"type": "url", "url": "https://example.com/native.png"},
    }
    null_url_block: dict[str, Any] = {"type": "image_url", "image_url": {"url": None}}
    empty_url_block: dict[str, Any] = {"type": "image_url", "image_url": {"url": ""}}
    non_b64_data_block: dict[str, Any] = {
        "type": "image_url",
        "image_url": {"url": "data:image/png,raw"},
    }
    messages: list[LLMMessage] = [
        {
            "role": "user",
            "content": [
                native_block,
                null_url_block,
                empty_url_block,
                non_b64_data_block,
            ],
        }
    ]

    assert llm.call(messages) == "ok"

    sent_content = sync_client.messages.create.call_args.kwargs["messages"][0][
        "content"
    ]
    assert sent_content == [
        native_block,
        null_url_block,
        empty_url_block,
        non_b64_data_block,
    ]


def test_get_input_tokens_count_converts_remote_image_url_blocks() -> None:
    llm, sync_client, _ = _build_llm()
    messages: list[LLMMessage] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Count tokens"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/count.png"},
                },
            ],
        }
    ]

    assert llm.get_input_tokens_count(messages) == 42

    counted_messages = sync_client.messages.count_tokens.call_args.kwargs["messages"]
    assert counted_messages[0]["content"][1] == {
        "type": "image",
        "source": {"type": "url", "url": "https://example.com/count.png"},
    }
