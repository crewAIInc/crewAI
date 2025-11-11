"""Tests for GPT-5 tool calling format normalization.

This module tests the handling of GPT-5's array-wrapped tool arguments format.
GPT-5 wraps arguments in an array like: [{"arg": "value"}, []]
while GPT-4 uses a flat dict: {"arg": "value"}
"""

from unittest.mock import MagicMock

import pytest
from crewai.tools.tool_usage import ToolUsage


def test_validate_tool_input_gpt5_wrapped_format():
    """Test that GPT-5's array-wrapped format is correctly normalized to a dict."""
    mock_agent = MagicMock()
    mock_agent.key = "test_agent_key"
    mock_agent.role = "test_agent_role"
    mock_agent._original_role = "test_agent_role"
    mock_agent.i18n = MagicMock()
    mock_agent.verbose = False

    mock_action = MagicMock()
    mock_action.tool = "test_tool"
    mock_action.tool_input = "test_input"

    tool_usage = ToolUsage(
        tools_handler=MagicMock(),
        tools=[],
        task=MagicMock(),
        function_calling_llm=None,
        agent=mock_agent,
        action=mock_action,
    )

    tool_input = '[{"responsible_employee_id": null, "include_inactive": false}, []]'
    expected_arguments = {"responsible_employee_id": None, "include_inactive": False}

    arguments = tool_usage._validate_tool_input(tool_input)
    assert arguments == expected_arguments


def test_validate_tool_input_gpt5_wrapped_format_single_element():
    """Test GPT-5 format with only the dict element (no trailing empty array)."""
    mock_agent = MagicMock()
    mock_agent.key = "test_agent_key"
    mock_agent.role = "test_agent_role"
    mock_agent._original_role = "test_agent_role"
    mock_agent.i18n = MagicMock()
    mock_agent.verbose = False

    mock_action = MagicMock()
    mock_action.tool = "test_tool"
    mock_action.tool_input = "test_input"

    tool_usage = ToolUsage(
        tools_handler=MagicMock(),
        tools=[],
        task=MagicMock(),
        function_calling_llm=None,
        agent=mock_agent,
        action=mock_action,
    )

    tool_input = '[{"key": "value", "number": 42}]'
    expected_arguments = {"key": "value", "number": 42}

    arguments = tool_usage._validate_tool_input(tool_input)
    assert arguments == expected_arguments


def test_validate_tool_input_gpt4_dict_format_unchanged():
    """Test that GPT-4's flat dict format continues to work unchanged."""
    tool_usage = ToolUsage(
        tools_handler=MagicMock(),
        tools=[],
        task=MagicMock(),
        function_calling_llm=None,
        agent=MagicMock(),
        action=MagicMock(),
    )

    tool_input = '{"responsible_employee_id": null, "include_inactive": false}'
    expected_arguments = {"responsible_employee_id": None, "include_inactive": False}

    arguments = tool_usage._validate_tool_input(tool_input)
    assert arguments == expected_arguments


def test_validate_tool_input_gpt5_wrapped_complex_args():
    """Test GPT-5 format with complex nested arguments."""
    mock_agent = MagicMock()
    mock_agent.key = "test_agent_key"
    mock_agent.role = "test_agent_role"
    mock_agent._original_role = "test_agent_role"
    mock_agent.i18n = MagicMock()
    mock_agent.verbose = False

    mock_action = MagicMock()
    mock_action.tool = "test_tool"
    mock_action.tool_input = "test_input"

    tool_usage = ToolUsage(
        tools_handler=MagicMock(),
        tools=[],
        task=MagicMock(),
        function_calling_llm=None,
        agent=mock_agent,
        action=mock_action,
    )

    tool_input = '[{"user": {"name": "Alice", "age": 30}, "items": [1, 2, 3]}, []]'
    expected_arguments = {
        "user": {"name": "Alice", "age": 30},
        "items": [1, 2, 3],
    }

    arguments = tool_usage._validate_tool_input(tool_input)
    assert arguments == expected_arguments


def test_validate_tool_input_invalid_list_format():
    """Test that invalid list formats (non-dict first element) are rejected."""
    # Create mock agent with proper string values
    mock_agent = MagicMock()
    mock_agent.key = "test_agent_key"
    mock_agent.role = "test_agent_role"
    mock_agent._original_role = "test_agent_role"
    mock_agent.i18n = MagicMock()
    mock_agent.verbose = False

    # Create mock action with proper string value
    mock_action = MagicMock()
    mock_action.tool = "test_tool"
    mock_action.tool_input = "test_input"

    tool_usage = ToolUsage(
        tools_handler=MagicMock(),
        tools=[],
        task=MagicMock(),
        function_calling_llm=None,
        agent=mock_agent,
        action=mock_action,
    )

    invalid_inputs = [
        '["string", "values"]',
        '[1, 2, 3]',
        '[null, {}]',
    ]

    for invalid_input in invalid_inputs:
        with pytest.raises(Exception) as e_info:
            tool_usage._validate_tool_input(invalid_input)
        assert (
            "Tool input must be a valid dictionary in JSON or Python literal format"
            in str(e_info.value)
        )


def test_validate_tool_input_gpt5_with_multiple_trailing_elements():
    """Test GPT-5 format with multiple trailing empty elements."""
    mock_agent = MagicMock()
    mock_agent.key = "test_agent_key"
    mock_agent.role = "test_agent_role"
    mock_agent._original_role = "test_agent_role"
    mock_agent.i18n = MagicMock()
    mock_agent.verbose = False

    mock_action = MagicMock()
    mock_action.tool = "test_tool"
    mock_action.tool_input = "test_input"

    tool_usage = ToolUsage(
        tools_handler=MagicMock(),
        tools=[],
        task=MagicMock(),
        function_calling_llm=None,
        agent=mock_agent,
        action=mock_action,
    )

    tool_input = '[{"key": "value"}, [], []]'
    expected_arguments = {"key": "value"}

    arguments = tool_usage._validate_tool_input(tool_input)
    assert arguments == expected_arguments
