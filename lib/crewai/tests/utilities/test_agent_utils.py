"""Tests for agent_utils module."""

import pytest

from crewai.agents.parser import AgentFinish, OutputParserError
from crewai.utilities.agent_utils import format_answer


class TestFormatAnswer:
    """Tests for the format_answer function."""

    def test_format_answer_with_valid_final_answer(self) -> None:
        """Test that format_answer correctly parses a valid final answer."""
        answer = """Thought: I have completed the task.
Final Answer: The result is 42."""

        result = format_answer(answer)

        assert isinstance(result, AgentFinish)
        assert result.output == "The result is 42."

    def test_format_answer_reraises_output_parser_error(self) -> None:
        """Test that format_answer re-raises OutputParserError for retry logic."""
        # Malformed output missing colons after "Thought", "Action", and "Action Input"
        malformed_answer = """Thought
The user wants to verify something.
Action
Video Analysis Tool
Action Input:
{"query": "Is there something?"}"""

        with pytest.raises(OutputParserError) as exc_info:
            format_answer(malformed_answer)

        # Verify that the error message contains helpful information
        assert exc_info.value.error is not None

    def test_format_answer_with_missing_action_colon(self) -> None:
        """Test that format_answer raises OutputParserError when Action colon is missing."""
        malformed_answer = """Thought: I need to search for information.
Action
Search Tool
Action Input: {"query": "test"}"""

        with pytest.raises(OutputParserError):
            format_answer(malformed_answer)

    def test_format_answer_with_missing_action_input_colon(self) -> None:
        """Test that format_answer raises OutputParserError when Action Input colon is missing."""
        malformed_answer = """Thought: I need to search for information.
Action: Search Tool
Action Input
{"query": "test"}"""

        with pytest.raises(OutputParserError):
            format_answer(malformed_answer)

    def test_format_answer_with_valid_action(self) -> None:
        """Test that format_answer correctly parses a valid action format."""
        valid_action = """Thought: I need to search for information.
Action: Search Tool
Action Input: {"query": "test"}"""

        # This should parse successfully without raising an exception
        result = format_answer(valid_action)

        # The result should be an AgentAction (not AgentFinish)
        assert result is not None
        assert not isinstance(result, AgentFinish)
