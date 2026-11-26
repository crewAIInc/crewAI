"""History replay must not crash on malformed tool-call arguments.

Assistant ``tool_calls`` arguments are stored verbatim from the LLM and can
be truncated (e.g. a stream cut by a length limit). The Gemini and OpenAI
providers guard the JSON parse when converting history; these tests lock the
same behavior in for ``_format_messages_for_anthropic``.
"""

from __future__ import annotations

from typing import Any

import pytest

from crewai.llms.providers.anthropic.completion import AnthropicCompletion


@pytest.fixture
def llm() -> AnthropicCompletion:
    return AnthropicCompletion(
        model="claude-3-5-sonnet-20241022", api_key="sk-dummy"
    )


def _history(arguments: Any) -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": arguments},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "search",
            "content": "72",
        },
    ]


def _format(llm: AnthropicCompletion, arguments: Any) -> list[dict[str, Any]]:
    formatted, _ = llm._format_messages_for_anthropic(_history(arguments))
    return formatted


def _assistant_tool_use(
    formatted: list[dict[str, Any]],
) -> dict[str, Any]:
    assistant = next(m for m in formatted if m.get("role") == "assistant")
    return assistant["content"][0]


def test_truncated_tool_arguments_do_not_crash(
    llm: AnthropicCompletion,
) -> None:
    tool_use = _assistant_tool_use(_format(llm, '{"query": "wea'))
    assert tool_use["type"] == "tool_use"
    assert tool_use["name"] == "search"
    assert tool_use["input"] == {}


def test_valid_tool_arguments_are_parsed(llm: AnthropicCompletion) -> None:
    tool_use = _assistant_tool_use(_format(llm, '{"query": "weather"}'))
    assert tool_use["input"] == {"query": "weather"}


def test_empty_tool_arguments_yield_empty_input(
    llm: AnthropicCompletion,
) -> None:
    tool_use = _assistant_tool_use(_format(llm, ""))
    assert tool_use["input"] == {}
