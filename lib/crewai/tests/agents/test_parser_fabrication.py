"""
Test for Issue #3154: Tool Fabrication Detection

This test verifies that the parser correctly detects when an LLM
fabricates tool execution by including both "Action:" and "Final Answer:"
in the same response.
"""
import pytest
from crewai.agents.parser import parse, OutputParserError


def test_parser_detects_fabricated_tool_execution():
    """
    Test that parser raises error when LLM fabricates tool execution.
    
    This is the bug from Issue #3154 where LLMs generate:
    - Thought
    - Action
    - Action Input
    - Observation (FABRICATED - tool never runs)
    - Final Answer
    
    All in one response, preventing tool.invoke() from ever being called.
    """
    fabricated_response = """Thought: I need to search for AI trends
Action: search_tool
Action Input: {"query": "AI trends"}
Observation: AI trends include large language models and multimodal AI.
Final Answer: Based on my research, the top AI trends are LLMs and multimodal AI."""
    
    with pytest.raises(OutputParserError) as exc_info:
        parse(fabricated_response)
    
    assert "both 'Action:' and 'Final Answer:'" in str(exc_info.value.error)
    assert "fabricated" in str(exc_info.value.error).lower()


def test_parser_allows_legitimate_action():
    """Test that parser still works correctly for legitimate actions."""
    legitimate_action = """Thought: I need to search
Action: search_tool
Action Input: {"query": "AI trends"}"""
    
    result = parse(legitimate_action)
    
    # Should return AgentAction, not raise error
    assert hasattr(result, 'tool')
    assert result.tool == 'search_tool'
    assert '"query": "AI trends"' in result.tool_input


def test_parser_allows_legitimate_final_answer():
    """Test that parser still works correctly for legitimate final answers."""
    legitimate_answer = """Thought: I have all the information
Final Answer: The answer is 42."""
    
    result = parse(legitimate_answer)
    
    # Should return AgentFinish, not raise error
    assert hasattr(result, 'output')
    assert result.output == "The answer is 42."
