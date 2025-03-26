import re
from unittest.mock import MagicMock, patch

import pytest

from crewai.agent import Agent
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource


def test_agent_with_chinese_role_name():
    """Test that an agent with a Chinese role name works correctly with the updated regex pattern."""
    # Create a knowledge source with some content
    content = "This is some test content."
    string_source = StringKnowledgeSource(content=content)
    
    # Mock the Knowledge class to avoid actual initialization
    with patch("crewai.agent.Knowledge") as MockKnowledge:
        mock_knowledge_instance = MockKnowledge.return_value
        
        # Create an agent with a Chinese role name
        agent = Agent(
            role="中文角色",  # Chinese role name
            goal="Test Chinese character support",
            backstory="Testing Chinese character support in agent role names.",
            knowledge_sources=[string_source],
        )
        
        # Call set_knowledge to trigger the regex pattern
        agent.set_knowledge()
        
        # Check that Knowledge was called with the correct collection_name
        calls = MockKnowledge.call_args_list
        for call in calls:
            args, kwargs = call
            if 'collection_name' in kwargs:
                collection_name = kwargs['collection_name']
                print(f"Collection name: {collection_name}")
                # The collection name should contain the Chinese characters
                assert "中文角色" == collection_name
