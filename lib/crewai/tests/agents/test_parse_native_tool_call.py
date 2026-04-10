"""Unit tests for _parse_native_tool_call in CrewAgentExecutor.

Verifies correct argument extraction for multiple LLM provider formats:
- OpenAI-style dicts (with "function" wrapper)
- Bedrock Converse API dicts (with "input" field, no "function" wrapper)
- Object-based tool calls (with attributes)

Regression tests for:
- https://github.com/crewAIInc/crewAI/issues/5275
- https://github.com/crewAIInc/crewAI/issues/4972
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from crewai.agents.crew_agent_executor import CrewAgentExecutor


@pytest.fixture()
def executor() -> CrewAgentExecutor:
    """Build a minimal CrewAgentExecutor for unit-testing _parse_native_tool_call."""
    exe = CrewAgentExecutor.model_construct()
    return exe


# ---------------------------------------------------------------------------
# Bedrock Converse API format (dict with "input", no "function" wrapper)
# Regression: issues #5275, #4972
# ---------------------------------------------------------------------------


class TestBedrockDictFormat:
    """Bedrock returns tool calls as dicts with 'name', 'input', 'toolUseId'."""

    def test_bedrock_basic_args(self, executor: CrewAgentExecutor) -> None:
        """Tool arguments in Bedrock 'input' field should be preserved."""
        tool_call: dict[str, Any] = {
            "name": "get_travel_details",
            "toolUseId": "tool_abc123",
            "input": {"city": "Paris"},
        }
        result = executor._parse_native_tool_call(tool_call)
        assert result is not None
        call_id, func_name, func_args = result
        assert call_id == "tool_abc123"
        assert func_name == "get_travel_details"
        assert func_args == {"city": "Paris"}

    def test_bedrock_multiple_args(self, executor: CrewAgentExecutor) -> None:
        """Multiple arguments should all be passed through."""
        tool_call: dict[str, Any] = {
            "name": "search_flights",
            "toolUseId": "tool_xyz789",
            "input": {
                "origin": "NYC",
                "destination": "LAX",
                "date": "2026-05-01",
            },
        }
        result = executor._parse_native_tool_call(tool_call)
        assert result is not None
        _, _, func_args = result
        assert func_args == {
            "origin": "NYC",
            "destination": "LAX",
            "date": "2026-05-01",
        }

    def test_bedrock_nested_objects(self, executor: CrewAgentExecutor) -> None:
        """Nested objects in Bedrock input should be preserved."""
        tool_call: dict[str, Any] = {
            "name": "create_booking",
            "toolUseId": "tool_nested",
            "input": {
                "passenger": {
                    "first_name": "Alice",
                    "last_name": "Smith",
                },
                "flights": ["FL100", "FL200"],
            },
        }
        result = executor._parse_native_tool_call(tool_call)
        assert result is not None
        _, _, func_args = result
        assert func_args["passenger"]["first_name"] == "Alice"
        assert func_args["flights"] == ["FL100", "FL200"]

    def test_bedrock_empty_input(self, executor: CrewAgentExecutor) -> None:
        """Empty input dict should be returned as-is (no-arg tools)."""
        tool_call: dict[str, Any] = {
            "name": "get_current_time",
            "toolUseId": "tool_empty",
            "input": {},
        }
        result = executor._parse_native_tool_call(tool_call)
        assert result is not None
        _, _, func_args = result
        # Empty input should fall through to default "{}" since {} is falsy
        # Either "{}" or {} is acceptable for no-arg tools
        assert func_args == "{}" or func_args == {}

    def test_bedrock_id_fallback(self, executor: CrewAgentExecutor) -> None:
        """When 'id' is absent, 'toolUseId' should be used."""
        tool_call: dict[str, Any] = {
            "name": "my_tool",
            "toolUseId": "bedrock_id_456",
            "input": {"query": "hello"},
        }
        result = executor._parse_native_tool_call(tool_call)
        assert result is not None
        call_id, _, _ = result
        assert call_id == "bedrock_id_456"

    def test_bedrock_numeric_args(self, executor: CrewAgentExecutor) -> None:
        """Numeric values in input should be preserved without coercion."""
        tool_call: dict[str, Any] = {
            "name": "calculate",
            "toolUseId": "tool_num",
            "input": {"x": 42, "y": 3.14, "negate": True},
        }
        result = executor._parse_native_tool_call(tool_call)
        assert result is not None
        _, _, func_args = result
        assert func_args["x"] == 42
        assert func_args["y"] == 3.14
        assert func_args["negate"] is True


# ---------------------------------------------------------------------------
# OpenAI-style dict format (with "function" wrapper and "arguments" as JSON string)
# ---------------------------------------------------------------------------


class TestOpenAIDictFormat:
    """OpenAI returns dicts with 'function.name' and 'function.arguments'."""

    def test_openai_basic_args(self, executor: CrewAgentExecutor) -> None:
        """Standard OpenAI dict format should extract arguments correctly."""
        tool_call: dict[str, Any] = {
            "id": "call_abc123",
            "function": {
                "name": "get_weather",
                "arguments": json.dumps({"location": "London"}),
            },
        }
        result = executor._parse_native_tool_call(tool_call)
        assert result is not None
        call_id, func_name, func_args = result
        assert call_id == "call_abc123"
        assert func_name == "get_weather"
        assert func_args == json.dumps({"location": "London"})

    def test_openai_empty_arguments(self, executor: CrewAgentExecutor) -> None:
        """OpenAI with empty arguments string should return that string."""
        tool_call: dict[str, Any] = {
            "id": "call_empty",
            "function": {
                "name": "get_time",
                "arguments": "{}",
            },
        }
        result = executor._parse_native_tool_call(tool_call)
        assert result is not None
        _, _, func_args = result
        # "{}" is truthy so it should be returned as-is
        assert func_args == "{}"

    def test_openai_no_id_generates_fallback(self, executor: CrewAgentExecutor) -> None:
        """Missing 'id' should generate a fallback call ID."""
        tool_call: dict[str, Any] = {
            "function": {
                "name": "some_tool",
                "arguments": '{"a": 1}',
            },
        }
        result = executor._parse_native_tool_call(tool_call)
        assert result is not None
        call_id, _, _ = result
        assert call_id.startswith("call_")

    def test_openai_complex_arguments(self, executor: CrewAgentExecutor) -> None:
        """Complex nested JSON arguments should be preserved."""
        args = {"filters": {"status": "active", "tags": ["urgent", "bug"]}, "limit": 10}
        tool_call: dict[str, Any] = {
            "id": "call_complex",
            "function": {
                "name": "search_issues",
                "arguments": json.dumps(args),
            },
        }
        result = executor._parse_native_tool_call(tool_call)
        assert result is not None
        _, _, func_args = result
        assert json.loads(func_args) == args


# ---------------------------------------------------------------------------
# Object-based tool calls (with attributes, e.g. litellm ModelResponse)
# ---------------------------------------------------------------------------


class TestObjectFormat:
    """Tool calls as objects with .function.name, .function.arguments attributes."""

    def test_object_with_function_attr(self, executor: CrewAgentExecutor) -> None:
        """Object with .function attribute should extract correctly."""
        func = SimpleNamespace(name="calculator", arguments='{"expr": "2+2"}')
        tool_call = SimpleNamespace(id="call_obj1", function=func)
        result = executor._parse_native_tool_call(tool_call)
        assert result is not None
        call_id, func_name, func_args = result
        assert call_id == "call_obj1"
        assert func_name == "calculator"
        assert func_args == '{"expr": "2+2"}'

    def test_object_with_name_and_input(self, executor: CrewAgentExecutor) -> None:
        """Object with .name and .input (Bedrock object format) should work."""
        tool_call = SimpleNamespace(
            id="call_bedrock_obj",
            name="search_tool",
            input={"search_query": "test query"},
        )
        # Remove 'function' attribute to ensure it doesn't match that branch
        result = executor._parse_native_tool_call(tool_call)
        assert result is not None
        call_id, func_name, func_args = result
        assert call_id == "call_bedrock_obj"
        assert func_name == "search_tool"
        assert func_args == {"search_query": "test query"}

    def test_object_with_function_call_attr(self, executor: CrewAgentExecutor) -> None:
        """Object with .function_call (Gemini-style) should extract correctly."""
        func_call = SimpleNamespace(name="my_tool", args={"key": "value"})
        tool_call = SimpleNamespace(function_call=func_call)
        # Ensure no .function attribute
        assert not hasattr(tool_call, "function") or tool_call.__dict__.get("function") is None
        result = executor._parse_native_tool_call(tool_call)
        assert result is not None
        _, func_name, func_args = result
        assert func_name == "my_tool"
        assert func_args == {"key": "value"}


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and unrecognized formats."""

    def test_unrecognized_object_returns_none(self, executor: CrewAgentExecutor) -> None:
        """An object without any known tool call attributes should return None."""
        tool_call = SimpleNamespace(foo="bar")
        result = executor._parse_native_tool_call(tool_call)
        assert result is None

    def test_dict_with_both_function_and_input(self, executor: CrewAgentExecutor) -> None:
        """When both 'function.arguments' and 'input' exist, arguments takes priority."""
        tool_call: dict[str, Any] = {
            "id": "call_both",
            "function": {
                "name": "dual_tool",
                "arguments": '{"from_function": true}',
            },
            "input": {"from_input": True},
        }
        result = executor._parse_native_tool_call(tool_call)
        assert result is not None
        _, _, func_args = result
        # function.arguments is truthy, so it should take priority
        assert func_args == '{"from_function": true}'

    def test_dict_with_function_wrapper_no_arguments(
        self, executor: CrewAgentExecutor
    ) -> None:
        """Dict with 'function' wrapper but no 'arguments' key should fall through to 'input'."""
        tool_call: dict[str, Any] = {
            "id": "call_no_args",
            "function": {"name": "partial_tool"},
            "input": {"fallback_key": "fallback_value"},
        }
        result = executor._parse_native_tool_call(tool_call)
        assert result is not None
        _, _, func_args = result
        assert func_args == {"fallback_key": "fallback_value"}

    def test_dict_no_function_no_input(self, executor: CrewAgentExecutor) -> None:
        """Dict with neither 'function.arguments' nor 'input' should default to '{}'."""
        tool_call: dict[str, Any] = {
            "name": "bare_tool",
        }
        result = executor._parse_native_tool_call(tool_call)
        assert result is not None
        _, func_name, func_args = result
        assert func_name == "bare_tool"
        assert func_args == "{}"

    def test_special_characters_in_args(self, executor: CrewAgentExecutor) -> None:
        """Arguments with special characters should be preserved."""
        tool_call: dict[str, Any] = {
            "name": "search",
            "toolUseId": "tool_special",
            "input": {"query": 'SELECT * FROM users WHERE name = "O\'Brien"'},
        }
        result = executor._parse_native_tool_call(tool_call)
        assert result is not None
        _, _, func_args = result
        assert func_args["query"] == 'SELECT * FROM users WHERE name = "O\'Brien"'
