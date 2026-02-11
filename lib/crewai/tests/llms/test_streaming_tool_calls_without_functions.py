"""Tests for streaming tool calls when available_functions is None.

When streaming is enabled and the LLM makes a tool call but available_functions
is not provided (None), the streaming methods should return the accumulated
tool calls as a list, allowing the caller (e.g., CrewAgentExecutor) to handle
tool execution externally.

This is the fix for issue #4442: Async stream does not work with function calls.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crewai.llm import LLM


@pytest.fixture
def get_calculator_tool_schema() -> dict[str, Any]:
    """Create a calculator tool schema for native function calling."""
    return {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression and return the result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate.",
                    }
                },
                "required": ["expression"],
            },
        },
    }


class TestStreamingToolCallsWithoutAvailableFunctions:
    """Tests for streaming mode when available_functions is None."""

    @pytest.mark.vcr()
    def test_sync_streaming_returns_tool_calls_without_available_functions(
        self, get_calculator_tool_schema: dict[str, Any]
    ) -> None:
        """Test that sync streaming returns tool calls when available_functions is None.

        This tests the fix for the bug where streaming with tools but without
        available_functions would return an empty string instead of the tool calls.
        """
        llm = LLM(model="openai/gpt-4o-mini", stream=True)

        result = llm.call(
            messages=[
                {"role": "user", "content": "Calculate 25 * 4 + 10 using the calculator"},
            ],
            tools=[get_calculator_tool_schema],
            available_functions=None,  # Key: no available_functions
        )

        # Result should be a list of tool calls, not an empty string
        assert result is not None, "Result should not be None"
        assert isinstance(result, list), f"Expected list of tool calls, got {type(result)}"
        assert len(result) > 0, "Should have at least one tool call"

        # Verify the tool call structure
        first_call = result[0]
        assert hasattr(first_call, "function") or (
            isinstance(first_call, dict) and "function" in first_call
        ), "Tool call should have function attribute/key"

        # Get function details
        if hasattr(first_call, "function"):
            func = first_call.function
            func_name = func.name
            func_args = func.arguments
        else:
            func = first_call["function"]
            func_name = func["name"]
            func_args = func["arguments"]

        assert func_name == "calculator", f"Expected 'calculator', got {func_name}"
        assert "expression" in func_args, "Arguments should contain 'expression'"

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_async_streaming_returns_tool_calls_without_available_functions(
        self, get_calculator_tool_schema: dict[str, Any]
    ) -> None:
        """Test that async streaming returns tool calls when available_functions is None.

        This is the main test for issue #4442.
        """
        llm = LLM(model="openai/gpt-4o-mini", stream=True)

        result = await llm.acall(
            messages=[
                {"role": "user", "content": "Calculate 25 * 4 + 10 using the calculator"},
            ],
            tools=[get_calculator_tool_schema],
            available_functions=None,  # Key: no available_functions
        )

        # Result should be a list of tool calls, not an empty string
        assert result is not None, "Result should not be None"
        assert isinstance(result, list), f"Expected list of tool calls, got {type(result)}"
        assert len(result) > 0, "Should have at least one tool call"

        # Verify the tool call structure
        first_call = result[0]
        assert hasattr(first_call, "function") or (
            isinstance(first_call, dict) and "function" in first_call
        ), "Tool call should have function attribute/key"

    @pytest.mark.vcr()
    def test_sync_streaming_still_works_with_available_functions(
        self, get_calculator_tool_schema: dict[str, Any]
    ) -> None:
        """Test that sync streaming still executes tools when available_functions is provided."""
        llm = LLM(model="openai/gpt-4o-mini", stream=True)

        def mock_calculator(expression: str) -> str:
            return f"Result: {eval(expression)}"

        result = llm.call(
            messages=[
                {"role": "user", "content": "Calculate 25 * 4 + 10 using the calculator"},
            ],
            tools=[get_calculator_tool_schema],
            available_functions={"calculator": mock_calculator},
        )

        # When available_functions is provided, tools are executed and result returned
        assert result is not None, "Result should not be None"
        # The result should contain the calculator output or be the final LLM response
        assert isinstance(result, str), f"Expected string result, got {type(result)}"

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_async_streaming_still_works_with_available_functions(
        self, get_calculator_tool_schema: dict[str, Any]
    ) -> None:
        """Test that async streaming still executes tools when available_functions is provided."""
        llm = LLM(model="openai/gpt-4o-mini", stream=True)

        def mock_calculator(expression: str) -> str:
            return f"Result: {eval(expression)}"

        result = await llm.acall(
            messages=[
                {"role": "user", "content": "Calculate 25 * 4 + 10 using the calculator"},
            ],
            tools=[get_calculator_tool_schema],
            available_functions={"calculator": mock_calculator},
        )

        # When available_functions is provided, tools are executed and result returned
        assert result is not None, "Result should not be None"
        assert isinstance(result, str), f"Expected string result, got {type(result)}"
