"""
Tests for verifying the integration of knowledge sources in the planning process.
This module ensures that agent knowledge is properly included during task planning.
"""

from unittest.mock import patch

import pytest

from crewai.agent import Agent
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from crewai.task import Task
from crewai.utilities.planning_handler import CrewPlanner


@pytest.fixture
def mock_knowledge_source():
    """
    Create a mock knowledge source with test content.
    Returns:
        StringKnowledgeSource:
            A knowledge source containing AI-related test content
    """
    content = """
    Important context about AI:
    1. AI systems use machine learning algorithms
    2. Neural networks are a key component
    3. Training data is essential for good performance
    """
    return StringKnowledgeSource(content=content)

@patch('crewai.knowledge.storage.knowledge_storage.chromadb')
def test_knowledge_included_in_planning(mock_chroma):
    """Test that verifies knowledge sources are properly included in planning."""
    # Mock ChromaDB collection
    mock_collection = mock_chroma.return_value.get_or_create_collection.return_value
    mock_collection.add.return_value = None
    
    # Create an agent with knowledge
    agent = Agent(
        role="AI Researcher",
        goal="Research and explain AI concepts",
        backstory="Expert in artificial intelligence",
        knowledge_sources=[
            StringKnowledgeSource(
                content="AI systems require careful training and validation."
            )
        ]
    )

    # Create a task for the agent
    task = Task(
        description="Explain the basics of AI systems",
        expected_output="A clear explanation of AI fundamentals",
        agent=agent
    )

    # Create a crew planner
    planner = CrewPlanner([task], None)

    # Get the task summary
    task_summary = planner._create_tasks_summary()

    # Verify that knowledge is included in planning when present
    assert "AI systems require careful training" in task_summary, \
        "Knowledge content should be present in task summary when knowledge exists"
    assert '"agent_knowledge"' in task_summary, \
        "agent_knowledge field should be present in task summary when knowledge exists"

    # Verify that knowledge is properly formatted
    assert isinstance(task.agent.knowledge_sources, list), \
        "Knowledge sources should be stored in a list"
    assert len(task.agent.knowledge_sources) > 0, \
        "At least one knowledge source should be present"
    assert task.agent.knowledge_sources[0].content in task_summary, \
        "Knowledge source content should be included in task summary"

    # Verify that other expected components are still present
    assert task.description in task_summary, \
        "Task description should be present in task summary"
    assert task.expected_output in task_summary, \
        "Expected output should be present in task summary"
    assert agent.role in task_summary, \
        "Agent role should be present in task summary"
