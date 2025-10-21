"""Test for crew reset_memories with readonly database error."""

from unittest.mock import MagicMock, patch

import pytest

from crewai.agent import Agent
from crewai.crew import Crew
from crewai.knowledge.knowledge import Knowledge
from crewai.process import Process
from crewai.task import Task


@pytest.fixture
def researcher():
    return Agent(
        role="Researcher",
        goal="Make the best research and analysis on content about AI and AI agents",
        backstory="You're an expert researcher, specialized in technology",
        allow_delegation=False,
    )


@pytest.fixture
def writer():
    return Agent(
        role="Senior Writer",
        goal="Write the best content about AI and AI agents.",
        backstory="You're a senior writer, specialized in technology",
        allow_delegation=False,
    )


def test_reset_all_memories_with_readonly_database_error_on_agent_knowledge(
    researcher, writer
):
    """Test that reset_memories('all') handles readonly database errors gracefully.
    
    This test simulates the exact scenario from issue #3753 where calling
    crew.reset_memories(command_type='all') fails with a "readonly database" error
    when trying to reset agent knowledge.
    """
    # Create mock knowledge objects for agents
    mock_ks_research = MagicMock(spec=Knowledge)
    mock_ks_writer = MagicMock(spec=Knowledge)
    
    # Simulate readonly database error when resetting agent knowledge
    mock_ks_research.reset.side_effect = Exception("attempt to write a readonly database")
    mock_ks_writer.reset.side_effect = Exception("attempt to write a readonly database")
    
    researcher.knowledge = mock_ks_research
    writer.knowledge = mock_ks_writer
    
    crew = Crew(
        agents=[researcher, writer],
        process=Process.sequential,
        tasks=[
            Task(description="Task 1", expected_output="output", agent=researcher),
            Task(description="Task 2", expected_output="output", agent=writer),
        ],
    )
    
    # This should not raise an exception - the readonly database error should be handled gracefully
    crew.reset_memories(command_type="all")
    
    # Verify that reset was attempted on both knowledge objects
    assert mock_ks_research.reset.call_count >= 1
    assert mock_ks_writer.reset.call_count >= 1


def test_reset_all_memories_with_collection_not_found_error_on_agent_knowledge(
    researcher, writer
):
    """Test that reset_memories('all') handles collection not found errors gracefully.
    
    Similar to readonly database errors, collection not found errors should also
    be handled gracefully as they indicate the collection is already reset.
    """
    # Create mock knowledge objects for agents
    mock_ks_research = MagicMock(spec=Knowledge)
    mock_ks_writer = MagicMock(spec=Knowledge)
    
    # Simulate collection not found error when resetting agent knowledge
    mock_ks_research.reset.side_effect = Exception("Collection does not exist")
    mock_ks_writer.reset.side_effect = Exception("Collection does not exist")
    
    researcher.knowledge = mock_ks_research
    writer.knowledge = mock_ks_writer
    
    crew = Crew(
        agents=[researcher, writer],
        process=Process.sequential,
        tasks=[
            Task(description="Task 1", expected_output="output", agent=researcher),
            Task(description="Task 2", expected_output="output", agent=writer),
        ],
    )
    
    # This should not raise an exception - the collection not found error should be handled gracefully
    crew.reset_memories(command_type="all")
    
    # Verify that reset was attempted on both knowledge objects
    assert mock_ks_research.reset.call_count >= 1
    assert mock_ks_writer.reset.call_count >= 1


def test_reset_agent_knowledge_with_readonly_database_error(researcher, writer):
    """Test that reset_memories('agent_knowledge') handles readonly database errors gracefully."""
    # Create mock knowledge objects for agents
    mock_ks_research = MagicMock(spec=Knowledge)
    mock_ks_writer = MagicMock(spec=Knowledge)
    
    # Simulate readonly database error when resetting agent knowledge
    mock_ks_research.reset.side_effect = Exception("attempt to write a readonly database")
    mock_ks_writer.reset.side_effect = Exception("attempt to write a readonly database")
    
    researcher.knowledge = mock_ks_research
    writer.knowledge = mock_ks_writer
    
    crew = Crew(
        agents=[researcher, writer],
        process=Process.sequential,
        tasks=[
            Task(description="Task 1", expected_output="output", agent=researcher),
            Task(description="Task 2", expected_output="output", agent=writer),
        ],
    )
    
    # This should not raise an exception - the readonly database error should be handled gracefully
    crew.reset_memories(command_type="agent_knowledge")
    
    # Verify that reset was attempted on both knowledge objects
    mock_ks_research.reset.assert_called_once()
    mock_ks_writer.reset.assert_called_once()


def test_reset_knowledge_with_readonly_database_error(researcher, writer):
    """Test that reset_memories('knowledge') handles readonly database errors gracefully."""
    # Create mock knowledge objects
    mock_ks_crew = MagicMock(spec=Knowledge)
    mock_ks_research = MagicMock(spec=Knowledge)
    mock_ks_writer = MagicMock(spec=Knowledge)
    
    # Simulate readonly database error when resetting knowledge
    mock_ks_crew.reset.side_effect = Exception("attempt to write a readonly database")
    mock_ks_research.reset.side_effect = Exception("attempt to write a readonly database")
    mock_ks_writer.reset.side_effect = Exception("attempt to write a readonly database")
    
    researcher.knowledge = mock_ks_research
    writer.knowledge = mock_ks_writer
    
    crew = Crew(
        agents=[researcher, writer],
        process=Process.sequential,
        tasks=[
            Task(description="Task 1", expected_output="output", agent=researcher),
            Task(description="Task 2", expected_output="output", agent=writer),
        ],
        knowledge=mock_ks_crew,
    )
    
    # This should not raise an exception - the readonly database error should be handled gracefully
    crew.reset_memories(command_type="knowledge")
    
    # Verify that reset was attempted on all knowledge objects
    mock_ks_crew.reset.assert_called_once()
    mock_ks_research.reset.assert_called_once()
    mock_ks_writer.reset.assert_called_once()


def test_reset_all_memories_with_unexpected_error_on_agent_knowledge(researcher, writer):
    """Test that reset_memories('all') propagates unexpected errors.
    
    Only readonly database and collection not found errors should be ignored.
    Other errors should be propagated.
    """
    # Create mock knowledge objects for agents
    mock_ks_research = MagicMock(spec=Knowledge)
    
    # Simulate an unexpected error
    mock_ks_research.reset.side_effect = Exception("Unexpected database error")
    
    researcher.knowledge = mock_ks_research
    
    crew = Crew(
        agents=[researcher, writer],
        process=Process.sequential,
        tasks=[
            Task(description="Task 1", expected_output="output", agent=researcher),
            Task(description="Task 2", expected_output="output", agent=writer),
        ],
    )
    
    # This should raise an exception because it's not a readonly database or collection not found error
    with pytest.raises(RuntimeError) as excinfo:
        crew.reset_memories(command_type="all")
    
    assert "Failed to reset" in str(excinfo.value)
    assert "Unexpected database error" in str(excinfo.value)
