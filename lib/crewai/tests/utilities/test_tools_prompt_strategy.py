"""Tests for tools_prompt_strategy configuration."""

from unittest.mock import Mock

import pytest

from crewai.utilities.agent_utils import get_tool_names, render_text_description_and_args


def test_get_tool_names_returns_comma_separated_names():
    """Test that get_tool_names returns comma-separated tool names."""
    tool1 = Mock()
    tool1.name = "search_tool"
    tool2 = Mock()
    tool2.name = "calculator_tool"
    tool3 = Mock()
    tool3.name = "file_reader_tool"
    
    tools = [tool1, tool2, tool3]
    result = get_tool_names(tools)
    
    assert result == "search_tool, calculator_tool, file_reader_tool"
    assert "description" not in result.lower()


def test_render_text_description_includes_descriptions():
    """Test that render_text_description_and_args includes full descriptions."""
    tool1 = Mock()
    tool1.description = "This is a search tool that searches the web for information"
    tool2 = Mock()
    tool2.description = "This is a calculator tool that performs mathematical operations"
    
    tools = [tool1, tool2]
    result = render_text_description_and_args(tools)
    
    assert "search tool" in result
    assert "calculator tool" in result
    assert "searches the web" in result
    assert "mathematical operations" in result


def test_names_only_strategy_is_shorter_than_full():
    """Test that names_only strategy produces shorter output than full descriptions."""
    tool1 = Mock()
    tool1.name = "search_tool"
    tool1.description = "This is a very long description " * 10
    tool2 = Mock()
    tool2.name = "calculator_tool"
    tool2.description = "This is another very long description " * 10
    
    tools = [tool1, tool2]
    
    names_only = get_tool_names(tools)
    full_description = render_text_description_and_args(tools)
    
    assert len(names_only) < len(full_description)
    assert len(names_only) < 100
    assert len(full_description) > 200


def test_agent_tools_prompt_strategy_config():
    """Test that Agent has tools_prompt_strategy configuration field."""
    from crewai.agent import Agent
    
    agent_full = Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory",
        tools_prompt_strategy="full",
    )
    
    agent_names = Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory",
        tools_prompt_strategy="names_only",
    )
    
    assert hasattr(agent_full, "tools_prompt_strategy")
    assert hasattr(agent_names, "tools_prompt_strategy")
    assert agent_full.tools_prompt_strategy == "full"
    assert agent_names.tools_prompt_strategy == "names_only"


def test_tools_prompt_strategy_default_is_full():
    """Test that tools_prompt_strategy defaults to 'full'."""
    from crewai.agent import Agent
    
    agent = Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory",
    )
    
    assert agent.tools_prompt_strategy == "full"
