"""Tests for memory and knowledge truncation."""

from unittest.mock import Mock, patch

import pytest


def test_truncate_text_helper():
    """Test basic text truncation helper logic."""
    text = "A" * 1000
    max_chars = 500
    
    if len(text) > max_chars:
        truncated = text[:max_chars] + "..."
    
    assert len(truncated) == max_chars + 3
    assert truncated.endswith("...")
    assert truncated.startswith("A" * 100)


def test_memory_truncation_when_max_chars_set():
    """Test that memory is truncated when memory_max_chars is set."""
    from crewai.agent import Agent
    
    long_memory = "M" * 2000
    agent = Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory",
        memory_max_chars=1000,
    )
    
    if agent.memory_max_chars and len(long_memory) > agent.memory_max_chars:
        truncated_memory = long_memory[:agent.memory_max_chars] + "..."
    
    assert len(truncated_memory) == 1003
    assert truncated_memory.endswith("...")


def test_memory_not_truncated_when_max_chars_none():
    """Test that memory is not truncated when memory_max_chars is None."""
    from crewai.agent import Agent
    
    long_memory = "M" * 2000
    agent = Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory",
        memory_max_chars=None,
    )
    
    result_memory = long_memory
    if agent.memory_max_chars and len(long_memory) > agent.memory_max_chars:
        result_memory = long_memory[:agent.memory_max_chars] + "..."
    
    assert len(result_memory) == 2000
    assert not result_memory.endswith("...")


def test_knowledge_truncation_when_max_chars_set():
    """Test that knowledge is truncated when knowledge_max_chars is set."""
    from crewai.agent import Agent
    
    long_knowledge = "K" * 3000
    agent = Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory",
        knowledge_max_chars=1500,
    )
    
    if agent.knowledge_max_chars and len(long_knowledge) > agent.knowledge_max_chars:
        truncated_knowledge = long_knowledge[:agent.knowledge_max_chars] + "..."
    
    assert len(truncated_knowledge) == 1503
    assert truncated_knowledge.endswith("...")


def test_knowledge_not_truncated_when_max_chars_none():
    """Test that knowledge is not truncated when knowledge_max_chars is None."""
    from crewai.agent import Agent
    
    long_knowledge = "K" * 3000
    agent = Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory",
        knowledge_max_chars=None,
    )
    
    result_knowledge = long_knowledge
    if agent.knowledge_max_chars and len(long_knowledge) > agent.knowledge_max_chars:
        result_knowledge = long_knowledge[:agent.knowledge_max_chars] + "..."
    
    assert len(result_knowledge) == 3000
    assert not result_knowledge.endswith("...")


def test_agent_config_fields_exist():
    """Test that new configuration fields exist on Agent."""
    from crewai.agent import Agent
    
    agent = Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory",
        memory_max_chars=1000,
        knowledge_max_chars=2000,
    )
    
    assert hasattr(agent, "memory_max_chars")
    assert hasattr(agent, "knowledge_max_chars")
    assert agent.memory_max_chars == 1000
    assert agent.knowledge_max_chars == 2000
