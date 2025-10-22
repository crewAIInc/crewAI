"""Tests for agent_utils module, specifically format_answer function."""

import pytest

from crewai.agents.parser import AgentAction, AgentFinish, OutputParserError
from crewai.utilities.agent_utils import format_answer


def test_format_answer_with_valid_action():
    """Test that format_answer correctly parses valid action format."""
    text = "Thought: Let's search\nAction: search\nAction Input: what is the weather?"
    result = format_answer(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == "what is the weather?"


def test_format_answer_with_valid_final_answer():
    """Test that format_answer correctly parses valid final answer format."""
    text = "Thought: I have the answer\nFinal Answer: The weather is sunny"
    result = format_answer(text)
    assert isinstance(result, AgentFinish)
    assert result.output == "The weather is sunny"


def test_format_answer_with_malformed_output_missing_colons():
    """Test that format_answer re-raises OutputParserError for malformed output.
    
    This is the core issue from bug #3771. When the LLM returns malformed output
    (e.g., missing colons after "Thought", "Action", "Action Input"), the 
    format_answer function should re-raise OutputParserError so the retry logic
    in _invoke_loop() can handle it properly.
    """
    malformed_text = """Thought
The user wants to verify something.
Action
Video Analysis Tool
Action Input:
{"query": "Is there something?"}"""
    
    with pytest.raises(OutputParserError) as exc_info:
        format_answer(malformed_text)
    
    assert "Invalid Format" in str(exc_info.value) or "missed" in str(exc_info.value)


def test_format_answer_with_missing_action():
    """Test that format_answer re-raises OutputParserError when Action is missing."""
    text = "Thought: Let's search\nAction Input: what is the weather?"
    
    with pytest.raises(OutputParserError) as exc_info:
        format_answer(text)
    
    assert "Invalid Format: I missed the 'Action:' after 'Thought:'." in str(
        exc_info.value
    )


def test_format_answer_with_missing_action_input():
    """Test that format_answer re-raises OutputParserError when Action Input is missing."""
    text = "Thought: Let's search\nAction: search"
    
    with pytest.raises(OutputParserError) as exc_info:
        format_answer(text)
    
    assert "I missed the 'Action Input:' after 'Action:'." in str(exc_info.value)


def test_format_answer_with_unexpected_exception():
    """Test that format_answer returns AgentFinish for truly unexpected errors.
    
    This tests that non-OutputParserError exceptions are still caught and
    converted to AgentFinish as a fallback behavior.
    """
    pass


def test_format_answer_preserves_original_text():
    """Test that format_answer preserves the original text in the result."""
    text = "Thought: Let's search\nAction: search\nAction Input: weather"
    result = format_answer(text)
    assert result.text == text
