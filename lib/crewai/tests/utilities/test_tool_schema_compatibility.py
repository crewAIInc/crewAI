"""Tests for tool schema compatibility across LLM providers.

Covers issue #4472: MCP tools producing JSON schemas that are incompatible
with Bedrock (Claude) and Gemini when using ``generate_model_description``
(OpenAI-specific) instead of the provider-agnostic
``generate_tool_parameters_schema``.
"""

from __future__ import annotations

from typing import Any, Optional

import pytest
from pydantic import BaseModel, Field

from crewai.llms.providers.utils.common import extract_tool_info, safe_tool_conversion
from crewai.tools.base_tool import BaseTool
from crewai.utilities.pydantic_schema_utils import (
    _strip_schema_metadata,
    generate_model_description,
    generate_tool_parameters_schema,
)


class FirewallToolInput(BaseModel):
    """Simulates an MCP tool input schema similar to the one in issue #4472."""

    hostname: str = Field(description="The firewall IP address")
    command: Optional[str] = Field(default=None, description="The CLI command string")


class SimpleToolInput(BaseModel):
    """A simple tool input with only required fields."""

    text: str = Field(description="Input text")


class ComplexToolInput(BaseModel):
    """Tool input with nested objects and arrays."""

    query: str = Field(description="Search query")
    max_results: int = Field(default=10, description="Maximum number of results")
    tags: list[str] = Field(default_factory=list, description="Filter tags")


class NestedModel(BaseModel):
    """A nested model for testing ref resolution."""

    name: str
    value: int


class ToolWithNestedInput(BaseModel):
    """Tool input containing a nested model for ref resolution."""

    config: NestedModel = Field(description="Configuration object")
    enabled: bool = Field(default=True, description="Whether enabled")


def _has_key_recursive(schema: Any, key: str) -> bool:
    """Check if a key exists anywhere in a nested dict/list structure."""
    if isinstance(schema, dict):
        if key in schema:
            return True
        return any(_has_key_recursive(v, key) for v in schema.values())
    if isinstance(schema, list):
        return any(_has_key_recursive(item, key) for item in schema)
    return False


class TestGenerateToolParametersSchema:
    """Tests for generate_tool_parameters_schema — the provider-agnostic path."""

    def test_no_title_field(self) -> None:
        schema = generate_tool_parameters_schema(FirewallToolInput)
        assert not _has_key_recursive(schema, "title")

    def test_no_additional_properties(self) -> None:
        schema = generate_tool_parameters_schema(FirewallToolInput)
        assert not _has_key_recursive(schema, "additionalProperties")

    def test_no_default_field(self) -> None:
        schema = generate_tool_parameters_schema(FirewallToolInput)
        assert not _has_key_recursive(schema, "default")

    def test_preserves_type(self) -> None:
        schema = generate_tool_parameters_schema(FirewallToolInput)
        assert schema["type"] == "object"

    def test_preserves_properties(self) -> None:
        schema = generate_tool_parameters_schema(FirewallToolInput)
        assert "hostname" in schema["properties"]
        assert "command" in schema["properties"]

    def test_preserves_required_only_required_fields(self) -> None:
        schema = generate_tool_parameters_schema(FirewallToolInput)
        assert "hostname" in schema.get("required", [])
        assert "command" not in schema.get("required", [])

    def test_preserves_description(self) -> None:
        schema = generate_tool_parameters_schema(FirewallToolInput)
        assert schema["properties"]["hostname"]["description"] == "The firewall IP address"

    def test_optional_field_type_is_string(self) -> None:
        """Optional fields should have null stripped and resolve to their base type."""
        schema = generate_tool_parameters_schema(FirewallToolInput)
        cmd = schema["properties"]["command"]
        assert cmd.get("type") == "string"
        assert "anyOf" not in cmd

    def test_simple_model(self) -> None:
        schema = generate_tool_parameters_schema(SimpleToolInput)
        assert schema["type"] == "object"
        assert "text" in schema["properties"]
        assert schema["properties"]["text"]["type"] == "string"
        assert "text" in schema.get("required", [])
        assert not _has_key_recursive(schema, "title")
        assert not _has_key_recursive(schema, "additionalProperties")

    def test_complex_model_with_array(self) -> None:
        schema = generate_tool_parameters_schema(ComplexToolInput)
        assert "tags" in schema["properties"]
        tags = schema["properties"]["tags"]
        assert tags["type"] == "array"
        assert tags["items"]["type"] == "string"
        assert not _has_key_recursive(schema, "title")
        assert not _has_key_recursive(schema, "default")

    def test_nested_model_refs_resolved(self) -> None:
        schema = generate_tool_parameters_schema(ToolWithNestedInput)
        assert not _has_key_recursive(schema, "$ref")
        assert "$defs" not in schema
        config_props = schema["properties"]["config"]
        assert "properties" in config_props
        assert "name" in config_props["properties"]
        assert "value" in config_props["properties"]


class TestStripSchemaMetadata:
    """Unit tests for the _strip_schema_metadata helper."""

    def test_strips_title(self) -> None:
        schema: dict[str, Any] = {"type": "object", "title": "Foo", "properties": {}}
        result = _strip_schema_metadata(schema)
        assert "title" not in result

    def test_strips_default(self) -> None:
        schema: dict[str, Any] = {"type": "string", "default": "bar"}
        result = _strip_schema_metadata(schema)
        assert "default" not in result

    def test_strips_additional_properties(self) -> None:
        schema: dict[str, Any] = {"type": "object", "additionalProperties": False}
        result = _strip_schema_metadata(schema)
        assert "additionalProperties" not in result

    def test_strips_nested(self) -> None:
        schema: dict[str, Any] = {
            "type": "object",
            "title": "Root",
            "properties": {
                "x": {"type": "string", "title": "X", "default": "a"},
                "nested": {
                    "type": "object",
                    "title": "Inner",
                    "additionalProperties": False,
                    "properties": {"y": {"type": "integer", "title": "Y"}},
                },
            },
        }
        result = _strip_schema_metadata(schema)
        assert not _has_key_recursive(result, "title")
        assert not _has_key_recursive(result, "default")
        assert not _has_key_recursive(result, "additionalProperties")

    def test_preserves_type_and_description(self) -> None:
        schema: dict[str, Any] = {
            "type": "string",
            "description": "hello",
            "title": "Foo",
        }
        result = _strip_schema_metadata(schema)
        assert result["type"] == "string"
        assert result["description"] == "hello"


class TestExtractToolInfoUsesCleanSchema:
    """Verify that extract_tool_info uses generate_tool_parameters_schema."""

    def test_args_schema_produces_clean_schema(self) -> None:
        tool_dict: dict[str, Any] = {
            "name": "firewall_tool",
            "description": "Run firewall commands",
            "args_schema": FirewallToolInput,
        }
        name, description, parameters = extract_tool_info(tool_dict)
        assert name == "firewall_tool"
        assert not _has_key_recursive(parameters, "title")
        assert not _has_key_recursive(parameters, "additionalProperties")
        assert not _has_key_recursive(parameters, "default")
        assert "hostname" in parameters.get("required", [])
        assert "command" not in parameters.get("required", [])

    def test_openai_format_uses_inline_parameters(self) -> None:
        tool_dict: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": "my_tool",
                "description": "desc",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                },
            },
        }
        name, description, parameters = extract_tool_info(tool_dict)
        assert name == "my_tool"
        assert parameters == {
            "type": "object",
            "properties": {"x": {"type": "string"}},
        }


class TestSafeToolConversionForProviders:
    """Integration tests verifying safe_tool_conversion produces provider-compatible schemas."""

    @staticmethod
    def _make_tool_dict(args_model: type[BaseModel]) -> dict[str, Any]:
        return {
            "name": "test_tool",
            "description": "A test tool",
            "args_schema": args_model,
        }

    def test_gemini_compatible_schema(self) -> None:
        _, _, params = safe_tool_conversion(
            self._make_tool_dict(FirewallToolInput), "Gemini"
        )
        assert not _has_key_recursive(params, "title")
        assert not _has_key_recursive(params, "additionalProperties")
        assert not _has_key_recursive(params, "default")
        assert params["properties"]["hostname"]["type"] == "string"
        assert params["properties"]["command"]["type"] == "string"

    def test_bedrock_compatible_schema(self) -> None:
        _, _, params = safe_tool_conversion(
            self._make_tool_dict(FirewallToolInput), "Bedrock"
        )
        assert not _has_key_recursive(params, "title")
        assert not _has_key_recursive(params, "additionalProperties")
        assert not _has_key_recursive(params, "default")
        assert "hostname" in params.get("required", [])
        assert "command" not in params.get("required", [])

    def test_anthropic_compatible_schema(self) -> None:
        _, _, params = safe_tool_conversion(
            self._make_tool_dict(ComplexToolInput), "Anthropic"
        )
        assert not _has_key_recursive(params, "title")
        assert not _has_key_recursive(params, "additionalProperties")
        assert params["properties"]["tags"]["type"] == "array"

    def test_nested_model_clean_for_all_providers(self) -> None:
        for provider in ("Gemini", "Bedrock", "Anthropic"):
            _, _, params = safe_tool_conversion(
                self._make_tool_dict(ToolWithNestedInput), provider
            )
            assert not _has_key_recursive(params, "title"), f"Failed for {provider}"
            assert not _has_key_recursive(
                params, "additionalProperties"
            ), f"Failed for {provider}"
            assert not _has_key_recursive(params, "$ref"), f"Failed for {provider}"


class TestOpenAISchemaUnchanged:
    """Ensure generate_model_description still produces OpenAI-specific schemas."""

    def test_has_additional_properties_false(self) -> None:
        result = generate_model_description(FirewallToolInput)
        schema = result["json_schema"]["schema"]
        assert schema.get("additionalProperties") is False

    def test_all_properties_required(self) -> None:
        result = generate_model_description(FirewallToolInput)
        schema = result["json_schema"]["schema"]
        assert "hostname" in schema["required"]
        assert "command" in schema["required"]
