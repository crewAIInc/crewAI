"""Test knowledge integration in planning process."""

import pytest
from unittest.mock import patch

from crewai.agent import Agent
from crewai.task import Task
from crewai.utilities.planning_handler import CrewPlanner
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource

@pytest.fixture
def mock_knowledge_source():
    """Create a mock knowledge source with test content."""
    content = """
    Important context about AI:
    1. AI systems use machine learning algorithms
    2. Neural networks are a key component
    3. Training data is essential for good performance
    """
    return StringKnowledgeSource(content=content)

def test_knowledge_not_in_planning():
    """Test that demonstrates knowledge sources are not included in planning."""
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

    # Verify that knowledge is not included
    assert "AI systems require careful training" not in task_summary
    assert '"agent_knowledge"' not in task_summary

    # Verify that other expected components are present
    assert task.description in task_summary
    assert task.expected_output in task_summary
    assert agent.role in task_summary
