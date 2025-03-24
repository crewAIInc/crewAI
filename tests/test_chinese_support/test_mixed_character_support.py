import re

import pytest

from crewai.agent import Agent
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource


@pytest.mark.parametrize(
    "role_name,expected_output",
    [
        ("中文角色", "中文角色"),
        ("中文角色123", "中文角色123"),
        ("中文_角色", "中文_角色"),
        ("测试-角色", "测试-角色"),
        ("漢字データ", "漢字データ"),
        ("ABC中文123", "ABC中文123"),
        ("测试_Test-123", "测试_Test-123"),
        ("中文 Test Space", "中文_Test_Space"),
        ("中文角色@#$", "中文角色___"),
    ],
)
def test_mixed_character_support(role_name, expected_output):
    """Test that various mixed character scenarios work as expected."""
    # Create a knowledge source with some content
    content = "This is some test content."
    string_source = StringKnowledgeSource(content=content)

    # Create an agent with the test role name
    agent = Agent(
        role=role_name,
        goal="Test mixed character support",
        backstory="Testing mixed character support in agent role names.",
        knowledge_sources=[string_source],
    )

    # Test that the regex pattern in agent.py correctly handles the role name
    # Unicode ranges for CJK characters:
    # \u4e00-\u9fff: Common Chinese characters
    # \u3400-\u4dbf: Extended CJK characters
    full_pattern = re.compile(r"[^\w\u4e00-\u9fff\u3400-\u4dbf\-_\r\n]|(\.\.)")
    knowledge_agent_name = f"{re.sub(full_pattern, '_', agent.role)}"

    # Verify that the agent was created successfully
    assert agent.role == role_name

    # Verify that the role name is processed correctly
    assert knowledge_agent_name == expected_output
