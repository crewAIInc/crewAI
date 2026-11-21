"""Tests for opt-in hosted tool search on the OpenAI Responses API path."""

from unittest.mock import MagicMock

from crewai.llm import LLM
from crewai.llms.providers.openai.completion import (
    OpenAICompletion,
    OpenAIToolSearchConfig,
)


def make_tool(name: str, **extra) -> dict:
    return {
        "name": name,
        "description": f"Run {name}",
        "parameters": {"type": "object", "properties": {}, "required": []},
        **extra,
    }


TOOLS = [make_tool("get_weather"), make_tool("get_time")]
MESSAGES = [{"role": "user", "content": "hi"}]


def build(**kwargs) -> OpenAICompletion:
    return OpenAICompletion(model="gpt-5.5", api_key="sk-test", api="responses", **kwargs)


def test_tool_search_is_disabled_by_default():
    llm = build()
    assert llm.tool_search is None

    tools = llm._prepare_responses_params(MESSAGES, tools=TOOLS)["tools"]

    assert [t["type"] for t in tools] == ["function", "function"]
    assert all("defer_loading" not in t for t in tools)


def test_tool_search_true_normalizes_to_config():
    assert build(tool_search=True).tool_search == OpenAIToolSearchConfig()
    assert build(tool_search=False).tool_search is None


def test_tool_search_via_llm_factory():
    llm = LLM(model="openai/gpt-5.5", api_key="sk-test", api="responses", tool_search=True)

    assert isinstance(llm, OpenAICompletion)
    assert llm.tool_search == OpenAIToolSearchConfig()


def test_tool_search_defers_function_tools():
    llm = build(tool_search=True)

    tools = llm._prepare_responses_params(MESSAGES, tools=TOOLS)["tools"]

    assert tools[0] == {"type": "tool_search"}
    assert [t["name"] for t in tools[1:]] == ["get_weather", "get_time"]
    assert all(t["defer_loading"] is True for t in tools[1:])


def test_tool_search_keeps_explicit_defer_loading():
    llm = build(tool_search=True)
    tools = [make_tool("get_weather", defer_loading=False), make_tool("get_time")]

    converted = llm._prepare_responses_params(MESSAGES, tools=tools)["tools"]

    by_name = {t["name"]: t for t in converted if t["type"] == "function"}
    assert by_name["get_weather"]["defer_loading"] is False
    assert by_name["get_time"]["defer_loading"] is True


def test_tool_search_skipped_for_single_tool():
    """With one tool there is nothing to search, so it stays eagerly loaded."""
    llm = build(tool_search=True)

    tools = llm._prepare_responses_params(MESSAGES, tools=TOOLS[:1])["tools"]

    assert tools == [llm._convert_tools_for_responses(TOOLS[:1])[0]]
    assert "defer_loading" not in tools[0]


def test_single_explicitly_deferred_tool_gets_tool_search():
    """A deferred tool is only valid alongside tool_search, even when alone."""
    llm = build(tool_search=True)

    tools = llm._prepare_responses_params(
        MESSAGES, tools=[make_tool("get_weather", defer_loading=True)]
    )["tools"]

    assert tools[0] == {"type": "tool_search"}
    assert tools[1]["name"] == "get_weather"
    assert tools[1]["defer_loading"] is True


def test_explicitly_deferred_tool_gets_tool_search_without_config():
    llm = build()
    tools = [make_tool("get_weather", defer_loading=True), make_tool("get_time")]

    converted = llm._prepare_responses_params(MESSAGES, tools=tools)["tools"]

    assert converted[0] == {"type": "tool_search"}
    by_name = {t["name"]: t for t in converted if t["type"] == "function"}
    assert by_name["get_weather"]["defer_loading"] is True
    # tool_search was not enabled, so tools without the flag stay eager
    assert "defer_loading" not in by_name["get_time"]


def test_tool_search_not_duplicated_when_listed_as_builtin():
    llm = build(tool_search=True, builtin_tools=["web_search", "tool_search"])

    tools = llm._prepare_responses_params(MESSAGES, tools=TOOLS)["tools"]

    types = [t["type"] for t in tools]
    assert types.count("tool_search") == 1
    assert types[0] == "web_search_preview"


def test_conversion_preserves_defer_loading_metadata():
    """Explicit flags survive conversion even when tool search is off."""
    llm = build()
    tools = [
        make_tool("top_level", defer_loading=True),
        {
            "type": "function",
            "function": {**make_tool("nested"), "defer_loading": True},
        },
        make_tool("plain"),
    ]

    converted = {t["name"]: t for t in llm._convert_tools_for_responses(tools)}

    assert converted["top_level"]["defer_loading"] is True
    assert converted["nested"]["defer_loading"] is True
    assert "defer_loading" not in converted["plain"]


def test_tool_search_request_sent_to_client():
    llm = build(tool_search=True)
    client = MagicMock()
    response = MagicMock(output=[], output_text="done", usage=None, id="resp_1")
    client.responses.create.return_value = response
    llm._client = client

    llm.call(MESSAGES, tools=TOOLS)

    sent = client.responses.create.call_args.kwargs["tools"]
    assert sent[0] == {"type": "tool_search"}
    assert all(t["defer_loading"] is True for t in sent[1:])
