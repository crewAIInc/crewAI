"""Tests for crewai.llms.providers.utils.common.

Focus: extract_tool_info / safe_tool_conversion with MCP-style tools that
use ``inputSchema`` instead of ``parameters``.

Regression tests for https://github.com/crewAIInc/crewAI/issues/4470.
"""

import pytest

from crewai.llms.providers.utils.common import extract_tool_info, safe_tool_conversion


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

STANDARD_OPENAI_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

MCP_INPUT_SCHEMA_TOOL = {
    "type": "function",
    "function": {
        "name": "list_resources",
        "description": "List MCP resources",
        # MCP servers send inputSchema, not parameters
        "inputSchema": {
            "type": "object",
            "properties": {"filter": {"type": "string"}},
            "required": [],
        },
    },
}

MCP_DIRECT_INPUT_SCHEMA_TOOL = {
    "name": "direct_mcp_tool",
    "description": "Direct-format MCP tool",
    "inputSchema": {
        "type": "object",
        "properties": {"arg1": {"type": "integer"}},
        "required": ["arg1"],
    },
}


# ---------------------------------------------------------------------------
# extract_tool_info tests
# ---------------------------------------------------------------------------


class TestExtractToolInfo:
    def test_standard_parameters_key(self):
        name, desc, params = extract_tool_info(STANDARD_OPENAI_TOOL)
        assert name == "get_weather"
        assert "city" in params["properties"]

    def test_mcp_input_schema_in_function_wrapper(self):
        """Tools with inputSchema inside function wrapper must not return empty params.

        This is the core regression case for issue #4470.
        """
        name, desc, params = extract_tool_info(MCP_INPUT_SCHEMA_TOOL)
        assert name == "list_resources"
        assert params, "params should not be empty when inputSchema is provided"
        assert "filter" in params.get("properties", {})

    def test_mcp_input_schema_direct_format(self):
        """Tools in direct format with inputSchema are handled correctly."""
        name, desc, params = extract_tool_info(MCP_DIRECT_INPUT_SCHEMA_TOOL)
        assert name == "direct_mcp_tool"
        assert params, "params should not be empty when inputSchema is provided"
        assert "arg1" in params.get("properties", {})

    def test_parameters_takes_precedence_over_input_schema(self):
        """If both parameters and inputSchema are present, parameters wins."""
        tool = {
            "type": "function",
            "function": {
                "name": "ambiguous_tool",
                "description": "Has both keys",
                "parameters": {"type": "object", "properties": {"a": {"type": "string"}}},
                "inputSchema": {"type": "object", "properties": {"b": {"type": "number"}}},
            },
        }
        _, _, params = extract_tool_info(tool)
        assert "a" in params.get("properties", {})
        assert "b" not in params.get("properties", {})

    def test_empty_tool_raises(self):
        with pytest.raises((ValueError, KeyError)):
            extract_tool_info("not-a-dict")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# safe_tool_conversion tests (end-to-end)
# ---------------------------------------------------------------------------


class TestSafeToolConversion:
    def test_mcp_tool_via_safe_conversion(self):
        """safe_tool_conversion must surface MCP tool parameters, not empty dict."""
        validated_name, desc, params = safe_tool_conversion(MCP_INPUT_SCHEMA_TOOL, "Bedrock")
        assert validated_name == "list_resources"
        assert params, "Bedrock tool inputSchema must not be empty after conversion"

    def test_standard_tool_via_safe_conversion(self):
        validated_name, desc, params = safe_tool_conversion(STANDARD_OPENAI_TOOL, "Bedrock")
        assert validated_name == "get_weather"
        assert "city" in params.get("properties", {})
