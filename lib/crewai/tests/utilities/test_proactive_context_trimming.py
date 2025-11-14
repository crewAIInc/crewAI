"""Tests for proactive context trimming."""

import pytest

from crewai.utilities.agent_utils import trim_messages_structurally


def test_trim_messages_structurally_keeps_system_message():
    """Test that trim_messages_structurally preserves system messages."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I'm doing well"},
    ]
    
    trim_messages_structurally(messages, keep_last_n=1, max_total_chars=100)
    
    system_messages = [msg for msg in messages if msg.get("role") == "system"]
    assert len(system_messages) == 1
    assert system_messages[0]["content"] == "You are a helpful assistant"


def test_trim_messages_structurally_keeps_last_n_pairs():
    """Test that trim_messages_structurally keeps last N message pairs."""
    messages = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "A" * 10000},
        {"role": "assistant", "content": "B" * 10000},
        {"role": "user", "content": "C" * 10000},
        {"role": "assistant", "content": "D" * 10000},
        {"role": "user", "content": "E" * 100},
        {"role": "assistant", "content": "F" * 100},
    ]
    
    trim_messages_structurally(messages, keep_last_n=1, max_total_chars=1000)
    
    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[1]["content"] == "E" * 100
    assert messages[2]["content"] == "F" * 100


def test_trim_messages_structurally_no_trim_when_under_limit():
    """Test that trim_messages_structurally doesn't trim when under limit."""
    messages = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]
    
    original_length = len(messages)
    trim_messages_structurally(messages, keep_last_n=3, max_total_chars=50000)
    
    assert len(messages) == original_length


def test_trim_messages_structurally_handles_empty_messages():
    """Test that trim_messages_structurally handles empty message list."""
    messages = []
    
    trim_messages_structurally(messages, keep_last_n=3, max_total_chars=1000)
    
    assert len(messages) == 0


def test_trim_messages_structurally_with_multiple_system_messages():
    """Test that trim_messages_structurally preserves all system messages."""
    messages = [
        {"role": "system", "content": "System 1"},
        {"role": "system", "content": "System 2"},
        {"role": "user", "content": "A" * 10000},
        {"role": "assistant", "content": "B" * 10000},
        {"role": "user", "content": "C" * 100},
        {"role": "assistant", "content": "D" * 100},
    ]
    
    trim_messages_structurally(messages, keep_last_n=1, max_total_chars=1000)
    
    system_messages = [msg for msg in messages if msg.get("role") == "system"]
    assert len(system_messages) == 2


def test_agent_proactive_context_trimming_config():
    """Test that Agent has proactive_context_trimming configuration field."""
    from crewai.agent import Agent
    
    agent_with_trimming = Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory",
        proactive_context_trimming=True,
    )
    
    agent_without_trimming = Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory",
        proactive_context_trimming=False,
    )
    
    assert hasattr(agent_with_trimming, "proactive_context_trimming")
    assert hasattr(agent_without_trimming, "proactive_context_trimming")
    assert agent_with_trimming.proactive_context_trimming is True
    assert agent_without_trimming.proactive_context_trimming is False


def test_proactive_context_trimming_default_is_false():
    """Test that proactive_context_trimming defaults to False."""
    from crewai.agent import Agent
    
    agent = Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory",
    )
    
    assert agent.proactive_context_trimming is False


def test_trim_messages_structurally_calculates_total_chars_correctly():
    """Test that trim_messages_structurally calculates total characters correctly."""
    messages = [
        {"role": "system", "content": "12345"},
        {"role": "user", "content": "67890"},
        {"role": "assistant", "content": "ABCDE"},
    ]
    
    total_chars = sum(len(str(msg.get("content", ""))) for msg in messages)
    assert total_chars == 15
    
    trim_messages_structurally(messages, keep_last_n=3, max_total_chars=20)
    assert len(messages) == 3
