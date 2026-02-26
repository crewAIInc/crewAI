"""Tests for memory tools (RecallMemoryTool / RememberTool).

Regression coverage for https://github.com/crewAIInc/crewAI/issues/4611
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from crewai.tools.memory_tools import (
    RecallMemorySchema,
    RecallMemoryTool,
    RememberTool,
    create_memory_tools,
)
from crewai.utilities.agent_utils import convert_tools_to_openai_schema


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_recall_tool() -> RecallMemoryTool:
    """Create a RecallMemoryTool with a mocked memory backend."""
    mock_memory = MagicMock()
    mock_memory.recall.return_value = []
    tools = create_memory_tools(mock_memory)
    # First tool is always RecallMemoryTool
    return next(t for t in tools if isinstance(t, RecallMemoryTool))


def _make_remember_tool() -> RememberTool:
    """Create a RememberTool with a mocked memory backend."""
    mock_memory = MagicMock()
    mock_record = MagicMock()
    mock_record.scope = "/general"
    mock_record.importance = 0.8
    mock_memory.remember.return_value = mock_record
    tools = create_memory_tools(mock_memory)
    return next(t for t in tools if isinstance(t, RememberTool))


# ---------------------------------------------------------------------------
# RecallMemoryTool -- regression for #4611
# ---------------------------------------------------------------------------


class TestRecallMemoryToolValidation:
    """Validate that RecallMemoryTool.run() raises a clear error when required
    arguments are missing, instead of the cryptic
    ``_run() missing 1 required positional argument: 'queries'``.
    """

    def test_run_with_no_args_raises_validation_error(self) -> None:
        """run() with zero arguments must raise ValueError, not TypeError.

        This is the exact scenario from issue #4611.
        """
        tool = _make_recall_tool()
        with pytest.raises(ValueError, match="validation failed"):
            tool.run()

    def test_run_with_valid_queries_succeeds(self) -> None:
        """run() with a proper queries list should succeed."""
        tool = _make_recall_tool()
        result = tool.run(queries=["what is the project deadline?"])
        assert "No relevant memories found" in result

    def test_run_with_all_args_succeeds(self) -> None:
        """run() with all arguments should succeed."""
        tool = _make_recall_tool()
        result = tool.run(
            queries=["deadline", "budget"],
            scope="/project/alpha",
            depth="shallow",
        )
        assert "No relevant memories found" in result

    def test_run_with_wrong_field_name_raises(self) -> None:
        """Passing 'query' (singular) instead of 'queries' should raise."""
        tool = _make_recall_tool()
        with pytest.raises(ValueError, match="validation failed"):
            tool.run(query="test")

    def test_run_with_queries_as_string_raises(self) -> None:
        """'queries' must be a list; a plain string should be rejected by the schema."""
        tool = _make_recall_tool()
        with pytest.raises(ValueError, match="validation failed"):
            tool.run(queries="single string instead of list")


# ---------------------------------------------------------------------------
# Native tool-calling path (convert_tools_to_openai_schema)
# ---------------------------------------------------------------------------


class TestMemoryToolsNativeToolCalling:
    """Ensure memory tools work correctly through the native tool-calling path
    used by convert_tools_to_openai_schema.
    """

    def test_recall_tool_schema_contains_queries_field(self) -> None:
        """The OpenAI schema for RecallMemoryTool should expose 'queries' as an array."""
        mock_memory = MagicMock()
        mock_memory.recall.return_value = []
        tools = create_memory_tools(mock_memory)
        openai_tools, _ = convert_tools_to_openai_schema(tools)

        recall_schema = next(
            s for s in openai_tools if s["function"]["name"] == "search_memory"
        )
        params = recall_schema["function"]["parameters"]
        assert "queries" in params["properties"]
        assert params["properties"]["queries"]["type"] == "array"

    def test_recall_via_available_functions_with_valid_args(self) -> None:
        """Calling the tool via available_functions dict (native path) should work."""
        mock_memory = MagicMock()
        mock_memory.recall.return_value = []
        tools = create_memory_tools(mock_memory)
        _, available_functions = convert_tools_to_openai_schema(tools)

        func = available_functions["search_memory"]
        result = func(queries=["test query"], scope=None, depth="shallow")
        assert "No relevant memories found" in result

    def test_recall_via_available_functions_with_empty_args_raises(self) -> None:
        """Calling with empty args (as when LLM sends {}) should raise ValueError."""
        mock_memory = MagicMock()
        mock_memory.recall.return_value = []
        tools = create_memory_tools(mock_memory)
        _, available_functions = convert_tools_to_openai_schema(tools)

        func = available_functions["search_memory"]
        with pytest.raises(ValueError, match="validation failed"):
            func()


# ---------------------------------------------------------------------------
# RememberTool sanity checks
# ---------------------------------------------------------------------------


class TestRememberToolValidation:
    """Basic validation tests for RememberTool."""

    def test_run_with_no_args_raises_validation_error(self) -> None:
        """run() with no arguments should raise ValueError, not TypeError."""
        tool = _make_remember_tool()
        with pytest.raises(ValueError, match="validation failed"):
            tool.run()

    def test_run_with_valid_args_succeeds(self) -> None:
        """run() with proper arguments should succeed."""
        tool = _make_remember_tool()
        result = tool.run(contents=["The project deadline is March 1st"])
        assert isinstance(result, str)
