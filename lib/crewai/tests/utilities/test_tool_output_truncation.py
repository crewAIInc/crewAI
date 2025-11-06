"""Tests for tool output truncation functionality."""

import pytest

from crewai.agents.parser import AgentAction, AgentFinish
from crewai.tools.tool_types import ToolResult
from crewai.utilities.agent_utils import (
    estimate_token_count,
    handle_agent_action_core,
    truncate_tool_output,
)


class TestEstimateTokenCount:
    """Tests for estimate_token_count function."""

    def test_empty_string(self):
        """Test token count estimation for empty string."""
        assert estimate_token_count("") == 0

    def test_short_string(self):
        """Test token count estimation for short string."""
        text = "Hello world"
        assert estimate_token_count(text) == len(text) // 4

    def test_long_string(self):
        """Test token count estimation for long string."""
        text = "a" * 10000
        assert estimate_token_count(text) == 2500


class TestTruncateToolOutput:
    """Tests for truncate_tool_output function."""

    def test_no_truncation_needed(self):
        """Test that small outputs are not truncated."""
        output = "Small output"
        result = truncate_tool_output(output, max_tokens=100)
        assert result == output
        assert "[Tool output truncated" not in result

    def test_truncation_applied(self):
        """Test that large outputs are truncated."""
        output = "a" * 20000
        result = truncate_tool_output(output, max_tokens=1000)
        assert len(result) < len(output)
        assert "[Tool output truncated" in result
        assert "showing first 1000" in result

    def test_truncation_message_format(self):
        """Test that truncation message has correct format."""
        output = "a" * 20000
        result = truncate_tool_output(output, max_tokens=1000, tool_name="search")
        assert "[Tool output truncated:" in result
        assert "Please refine your query" in result

    def test_very_small_max_tokens(self):
        """Test truncation with very small max_tokens."""
        output = "a" * 1000
        result = truncate_tool_output(output, max_tokens=10)
        assert "[Tool output truncated" in result

    def test_exact_boundary(self):
        """Test truncation at exact token boundary."""
        output = "a" * 400
        result = truncate_tool_output(output, max_tokens=100)
        assert result == output


class TestHandleAgentActionCore:
    """Tests for handle_agent_action_core with tool output truncation."""

    def test_small_tool_output_not_truncated(self):
        """Test that small tool outputs are not truncated."""
        formatted_answer = AgentAction(
            text="Thought: I need to search",
            tool="search",
            tool_input={"query": "test"},
            thought="I need to search",
        )
        tool_result = ToolResult(result="Small result", result_as_answer=False)

        result = handle_agent_action_core(
            formatted_answer=formatted_answer,
            tool_result=tool_result,
            max_tool_output_tokens=1000,
        )

        assert isinstance(result, AgentAction)
        assert "Small result" in result.text
        assert "[Tool output truncated" not in result.text

    def test_large_tool_output_truncated(self):
        """Test that large tool outputs are truncated."""
        formatted_answer = AgentAction(
            text="Thought: I need to search",
            tool="search",
            tool_input={"query": "test"},
            thought="I need to search",
        )
        large_output = "a" * 20000
        tool_result = ToolResult(result=large_output, result_as_answer=False)

        result = handle_agent_action_core(
            formatted_answer=formatted_answer,
            tool_result=tool_result,
            max_tool_output_tokens=1000,
        )

        assert isinstance(result, AgentAction)
        assert "[Tool output truncated" in result.text
        assert len(result.result) < len(large_output)

    def test_truncation_with_result_as_answer(self):
        """Test that truncation works with result_as_answer=True."""
        formatted_answer = AgentAction(
            text="Thought: I need to search",
            tool="search",
            tool_input={"query": "test"},
            thought="I need to search",
        )
        large_output = "a" * 20000
        tool_result = ToolResult(result=large_output, result_as_answer=True)

        result = handle_agent_action_core(
            formatted_answer=formatted_answer,
            tool_result=tool_result,
            max_tool_output_tokens=1000,
        )

        assert isinstance(result, AgentFinish)
        assert "[Tool output truncated" in result.output
        assert len(result.output) < len(large_output)

    def test_custom_max_tokens(self):
        """Test that custom max_tool_output_tokens is respected."""
        formatted_answer = AgentAction(
            text="Thought: I need to search",
            tool="search",
            tool_input={"query": "test"},
            thought="I need to search",
        )
        large_output = "a" * 10000
        tool_result = ToolResult(result=large_output, result_as_answer=False)

        result = handle_agent_action_core(
            formatted_answer=formatted_answer,
            tool_result=tool_result,
            max_tool_output_tokens=500,
        )

        assert isinstance(result, AgentAction)
        assert "[Tool output truncated" in result.text
        assert "showing first 500" in result.text

    def test_step_callback_called(self):
        """Test that step_callback is called even with truncation."""
        formatted_answer = AgentAction(
            text="Thought: I need to search",
            tool="search",
            tool_input={"query": "test"},
            thought="I need to search",
        )
        tool_result = ToolResult(result="a" * 20000, result_as_answer=False)

        callback_called = []

        def step_callback(result):
            callback_called.append(result)

        handle_agent_action_core(
            formatted_answer=formatted_answer,
            tool_result=tool_result,
            step_callback=step_callback,
            max_tool_output_tokens=1000,
        )

        assert len(callback_called) == 1
        assert callback_called[0] == tool_result

    def test_show_logs_called(self):
        """Test that show_logs is called even with truncation."""
        formatted_answer = AgentAction(
            text="Thought: I need to search",
            tool="search",
            tool_input={"query": "test"},
            thought="I need to search",
        )
        tool_result = ToolResult(result="a" * 20000, result_as_answer=False)

        logs_called = []

        def show_logs(answer):
            logs_called.append(answer)

        handle_agent_action_core(
            formatted_answer=formatted_answer,
            tool_result=tool_result,
            show_logs=show_logs,
            max_tool_output_tokens=1000,
        )

        assert len(logs_called) == 1
        assert isinstance(logs_called[0], AgentAction)
