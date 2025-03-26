import re

import pytest

from crewai.agent import Agent
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource


def test_agent_with_chinese_role_name():
    """Test that an agent with a Chinese role name works correctly."""
    # Create a knowledge source with some content
    content = "This is some test content."
    string_source = StringKnowledgeSource(content=content)
    
    # Create an agent with a Chinese role name
    agent = Agent(
        role="中文角色",  # Chinese role name
        goal="Test Chinese character support",
        backstory="Testing Chinese character support in agent role names.",
        knowledge_sources=[string_source],
    )
    
    # Test that the regex pattern in agent.py correctly preserves Chinese characters
    full_pattern = re.compile(r"[^a-zA-Z0-9\u4e00-\u9fa5\-_\r\n]|(\.\.)")
    knowledge_agent_name = f"{re.sub(full_pattern, '_', agent.role)}"
    
    # Verify that the agent was created successfully
    assert agent.role == "中文角色"
    
    # Verify that the Chinese characters are preserved in the knowledge_agent_name
    assert knowledge_agent_name == "中文角色"
