import pytest
from unittest.mock import patch, MagicMock

from crewai.memory.short_term.short_term_memory import ShortTermMemory
from crewai.memory.short_term.short_term_memory_item import ShortTermMemoryItem
from crewai.agent import Agent
from crewai.crew import Crew
from crewai.task import Task


@pytest.fixture
def short_term_memory():
    """Fixture to create a ShortTermMemory instance"""
    agent = Agent(
        role="Researcher",
        goal="Search relevant data and provide results",
        backstory="You are a researcher at a leading tech think tank.",
        tools=[],
        verbose=True,
    )

    task = Task(
        description="Perform a search on specific topics.",
        expected_output="A list of relevant URLs based on the search query.",
        agent=agent,
    )
    return ShortTermMemory(crew=Crew(agents=[agent], tasks=[task]))


def test_save_with_custom_key(short_term_memory):
    """Test that save method correctly passes custom_key to storage"""
    with patch.object(short_term_memory.storage, 'save') as mock_save:
        short_term_memory.save(
            value="Test data",
            metadata={"task": "test_task"},
            agent="test_agent",
            custom_key="user123",
        )
        
        called_args = mock_save.call_args[0]
        called_kwargs = mock_save.call_args[1]
        
        assert "custom_key" in called_args[1]
        assert called_args[1]["custom_key"] == "user123"


def test_search_with_custom_key(short_term_memory):
    """Test that search method correctly passes custom_key to storage"""
    expected_results = [{"context": "Test data", "metadata": {"custom_key": "user123"}, "score": 0.95}]
    
    with patch.object(short_term_memory.storage, 'search', return_value=expected_results) as mock_search:
        results = short_term_memory.search("test query", custom_key="user123")
        
        mock_search.assert_called_once()
        filter_arg = mock_search.call_args[1].get('filter')
        assert filter_arg == {"custom_key": {"$eq": "user123"}}
        assert results == expected_results
