"""Contract tests for `_parse_native_tool_call`.

Locks in the behavior of each native tool-call shape CrewAgentExecutor supports
so provider-specific argument-passing bugs (e.g. #4972 — Bedrock Converse args
silently dropped) can't silently regress.

Each test asserts the parsed tuple `(call_id, func_name, func_args)` matches the
expected wire-format contract for that provider, before any pydantic validation
or tool dispatch runs.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from crewai.agents.crew_agent_executor import CrewAgentExecutor


def _parse(tool_call):
    """Invoke the bound method on a dummy self (the function does not touch self)."""
    return CrewAgentExecutor._parse_native_tool_call(MagicMock(), tool_call)


def test_openai_style_object():
    """OpenAI: tool_call.function.arguments is a JSON string passed through verbatim."""
    tc = SimpleNamespace(
        id="call_abc",
        function=SimpleNamespace(name="search", arguments='{"query": "hello"}'),
    )
    result = _parse(tc)
    assert result == ("call_abc", "search", '{"query": "hello"}')


def test_openai_style_object_without_id_synthesizes_one():
    tc = SimpleNamespace(function=SimpleNamespace(name="x", arguments="{}"))
    call_id, func_name, func_args = _parse(tc)
    assert call_id.startswith("call_")
    assert func_name == "x"
    assert func_args == "{}"


def test_gemini_style_function_call_object():
    """Gemini: tool_call.function_call.args is a Struct-like mapping coerced to dict."""
    tc = SimpleNamespace(
        function_call=SimpleNamespace(name="lookup", args={"q": "hi"})
    )
    call_id, func_name, func_args = _parse(tc)
    assert call_id.startswith("call_")
    assert func_name == "lookup"
    assert func_args == {"q": "hi"}


def test_gemini_style_with_empty_args_yields_empty_dict():
    tc = SimpleNamespace(function_call=SimpleNamespace(name="x", args=None))
    _, _, func_args = _parse(tc)
    assert func_args == {}


def test_anthropic_style_object():
    """Anthropic: tool_call.name + tool_call.input on the call object itself."""
    tc = SimpleNamespace(id="toolu_01ABC", name="search", input={"query": "claude"})
    result = _parse(tc)
    assert result == ("toolu_01ABC", "search", {"query": "claude"})


def test_dict_openai_format():
    """Dict variant with the OpenAI-style nested function field."""
    tc = {
        "id": "call_42",
        "function": {"name": "lookup", "arguments": '{"k": "v"}'},
    }
    result = _parse(tc)
    assert result == ("call_42", "lookup", '{"k": "v"}')


def test_dict_bedrock_converse_format():
    """Regression for #4972: Bedrock Converse dict carries args under 'input', not 'function.arguments'.

    Earlier versions used a truthy default that swallowed `input` and returned `{}`.
    The dict branch must read `tool_call['input']` whenever the function-side is empty.
    """
    tc = {
        "toolUseId": "tooluse_xyz",
        "name": "search_in_user_knowledgebase",
        "input": {"search_query": "test"},
    }
    call_id, func_name, func_args = _parse(tc)
    assert call_id == "tooluse_xyz"
    assert func_name == "search_in_user_knowledgebase"
    assert func_args == {"search_query": "test"}, (
        f"Bedrock args dropped: got {func_args!r}, expected {{'search_query': 'test'}}"
    )


def test_dict_bedrock_with_empty_input_dict_stays_empty():
    """Edge case: explicit empty dict from Bedrock should not be replaced by anything."""
    tc = {"toolUseId": "id1", "name": "noop", "input": {}}
    _, _, func_args = _parse(tc)
    assert func_args == {}


def test_dict_missing_input_and_function_returns_empty_args():
    """No arguments anywhere — should return empty dict, not raise."""
    tc = {"id": "x", "name": "noop"}
    _, _, func_args = _parse(tc)
    assert func_args == {}


def test_unrecognized_shape_returns_none():
    """Plain object with no recognized fields signals 'not a native tool call'."""
    assert _parse(SimpleNamespace(unrelated="data")) is None


@pytest.mark.parametrize(
    "tool_call,expected_args",
    [
        # OpenAI dict
        ({"id": "1", "function": {"name": "f", "arguments": "{}"}}, "{}"),
        # Bedrock dict
        ({"toolUseId": "1", "name": "f", "input": {"a": 1}}, {"a": 1}),
        # Mixed: function present but empty, input has the real args (Bedrock-via-OpenAI shim)
        ({"name": "f", "function": {}, "input": {"a": 2}}, {"a": 2}),
    ],
)
def test_dict_variants_preserve_args(tool_call, expected_args):
    _, _, func_args = _parse(tool_call)
    assert func_args == expected_args
