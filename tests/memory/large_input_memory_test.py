import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from crewai.memory.short_term.short_term_memory import ShortTermMemory
from crewai.agent import Agent
from crewai.crew import Crew
from crewai.task import Task
from crewai.utilities.constants import MEMORY_CHUNK_SIZE


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


def test_memory_with_large_input(short_term_memory):
    """Test that memory can handle large inputs without token limit errors"""
    large_input = "test value " * (MEMORY_CHUNK_SIZE + 1000)
    
    with patch.object(
        short_term_memory.storage, '_chunk_text', 
        return_value=["chunk1", "chunk2"]
    ) as mock_chunk_text:
        with patch.object(
            short_term_memory.storage.collection, 'add'
        ) as mock_add:
            short_term_memory.save(value=large_input, agent="test_agent")
            
            assert mock_chunk_text.called
        
    with patch.object(
        short_term_memory.storage, 'search',
        return_value=[{"context": large_input, "metadata": {"agent": "test_agent"}, "score": 0.95}]
    ):
        result = short_term_memory.search(large_input[:100], score_threshold=0.01)
        assert result[0]["context"] == large_input
        assert result[0]["metadata"]["agent"] == "test_agent"
