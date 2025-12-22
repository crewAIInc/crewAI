"""Tests for agent utility functions."""

import pytest

from crewai.agents.parser import AgentFinish
from crewai.utilities.agent_utils import _clean_raw_output, format_answer


class TestCleanRawOutput:
    """Tests for _clean_raw_output function."""

    def test_extracts_final_answer_when_present(self):
        """Test that Final Answer content is properly extracted."""
        answer = """Thought: I need to process this request.
Action: search
Action Input: {"query": "test"}
Observation: search results here
Thought: Now I have the answer.
Final Answer: The search returned positive results."""

        result = _clean_raw_output(answer)
        assert result == "The search returned positive results."

    def test_removes_thought_prefix(self):
        """Test that Thought: prefix lines are removed."""
        answer = """Thought: I'm thinking about the problem.
This is the actual content.
More content here."""

        result = _clean_raw_output(answer)
        assert "Thought:" not in result
        assert "This is the actual content." in result

    def test_removes_action_lines(self):
        """Test that Action: and Action Input: lines are removed."""
        answer = """Some content here.
Action: tool_name
Action Input: {"param": "value"}
More content after."""

        result = _clean_raw_output(answer)
        assert "Action:" not in result
        assert "Action Input:" not in result
        assert "Some content here." in result

    def test_removes_observation_lines(self):
        """Test that Observation: lines are removed."""
        answer = """Content before.
Observation: tool output here
Content after observation."""

        result = _clean_raw_output(answer)
        assert "Observation:" not in result
        assert "Content before." in result

    def test_returns_original_if_no_content_left(self):
        """Test that original is returned if cleaning removes everything."""
        answer = """Thought: Only thought here
Action: some_action"""

        result = _clean_raw_output(answer)
        # When cleaning results in empty content, return original
        assert result == answer

    def test_handles_plain_text(self):
        """Test that plain text without markers is returned as-is."""
        answer = "This is a simple response without any markers."
        result = _clean_raw_output(answer)
        assert result == answer

    def test_handles_multiline_final_answer(self):
        """Test that multiline Final Answer is properly extracted."""
        answer = """Thought: Processing...
Final Answer: This is line one.
This is line two.
And line three."""

        result = _clean_raw_output(answer)
        assert "This is line one." in result
        assert "This is line two." in result
        assert "And line three." in result


class TestFormatAnswer:
    """Tests for format_answer function."""

    def test_returns_agent_finish_on_parse_failure(self):
        """Test that AgentFinish is returned when parsing fails."""
        # Invalid format that will fail parsing
        answer = """Thought: Some thought here
This is not a valid format."""

        result = format_answer(answer)
        assert isinstance(result, AgentFinish)
        assert result.thought == "Failed to parse LLM response"

    def test_cleans_output_on_parse_failure(self):
        """Test that output is cleaned when parsing fails."""
        answer = """Thought: I need to respond.
Action: invalid_action
The actual response content here."""

        result = format_answer(answer)
        assert isinstance(result, AgentFinish)
        # The cleaned output should not contain internal markers
        assert "Thought:" not in result.output
        assert "Action:" not in result.output

    def test_preserves_original_text(self):
        """Test that original text is preserved in the text field."""
        answer = """Thought: Some thought.
Action: tool
The response."""

        result = format_answer(answer)
        assert isinstance(result, AgentFinish)
        # Original text should be preserved
        assert result.text == answer

    def test_valid_final_answer_format(self):
        """Test that valid Final Answer format is properly parsed."""
        answer = """Thought: I have the answer.
Final Answer: This is the correct response."""

        result = format_answer(answer)
        assert isinstance(result, AgentFinish)
        assert result.output == "This is the correct response."
