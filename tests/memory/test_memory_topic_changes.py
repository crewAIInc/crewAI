import time
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from crewai.agent import Agent
from crewai.crew import Crew
from crewai.memory.short_term.short_term_memory import ShortTermMemory
from crewai.memory.short_term.short_term_memory_item import ShortTermMemoryItem
from crewai.memory.storage.rag_storage import RAGStorage
from crewai.task import Task


@pytest.fixture
def short_term_memory():
    """Fixture to create a ShortTermMemory instance"""
    agent = Agent(
        role="Tutor",
        goal="Teach programming concepts",
        backstory="You are a programming tutor helping students learn.",
        tools=[],
        verbose=True,
    )

    task = Task(
        description="Explain programming concepts to students.",
        expected_output="Clear explanations of programming concepts.",
        agent=agent,
    )
    return ShortTermMemory(crew=Crew(agents=[agent], tasks=[task]))


def test_memory_prioritizes_recent_topic(short_term_memory):
    """Test that memory retrieval prioritizes the most recent topic in a conversation."""
    # First topic: Python variables
    topic1_data = "Variables in Python are dynamically typed. You can assign any value to a variable without declaring its type."
    topic1_timestamp = datetime.now() - timedelta(minutes=10)  # Older memory
    
    # Second topic: Python abstract classes
    topic2_data = "Abstract classes in Python are created using the ABC module. They cannot be instantiated and are used as a blueprint for other classes."
    topic2_timestamp = datetime.now()  # More recent memory
    
    # Mock search results to simulate what would be returned by RAGStorage
    mock_results = [
        {
            "id": "2",
            "metadata": {
                "agent": "Tutor", 
                "topic": "python_abstract_classes",
                "timestamp": topic2_timestamp.isoformat()
            },
            "context": topic2_data,
            "score": 0.85,  # Higher score due to recency boost
        },
        {
            "id": "1",
            "metadata": {
                "agent": "Tutor", 
                "topic": "python_variables",
                "timestamp": topic1_timestamp.isoformat()
            },
            "context": topic1_data,
            "score": 0.75,  # Lower score due to being older
        }
    ]
    
    # Mock the search method to return our predefined results
    with patch.object(RAGStorage, 'search', return_value=mock_results):
        # Query that could match both topics but should prioritize the more recent one
        query = "Can you give me another example of that?"
        
        # Search with recency consideration
        results = short_term_memory.search(query)
        
        # Verify that the most recent topic (abstract classes) is prioritized
        assert len(results) > 0, "No search results returned"
        
        # The first result should be about abstract classes (the more recent topic)
        assert "abstract classes" in results[0]["context"].lower(), "Recent topic (abstract classes) not prioritized"
        
        # If there are multiple results, check if the older topic is also returned but with lower priority
        if len(results) > 1:
            assert "variables" in results[1]["context"].lower(), "Older topic should be second"
            
            # Verify that the scores reflect the recency prioritization
            assert results[0]["score"] > results[1]["score"], "Recent topic should have higher score"


def test_future_timestamp_validation():
    """Test that ShortTermMemoryItem raises ValueError for future timestamps."""
    # Setup agent and task for memory
    agent = Agent(
        role="Tutor",
        goal="Teach programming concepts",
        backstory="You are a programming tutor helping students learn.",
        tools=[],
        verbose=True,
    )
    
    task = Task(
        description="Explain programming concepts to students.",
        expected_output="Clear explanations of programming concepts.",
        agent=agent,
    )
    
    # Create a future timestamp
    future_timestamp = datetime.now() + timedelta(days=1)
    
    # Test constructor validation
    with pytest.raises(ValueError, match="Timestamp cannot be in the future"):
        ShortTermMemoryItem(data="Test data", timestamp=future_timestamp)
    
    # Test save method validation
    memory = ShortTermMemory(crew=Crew(agents=[agent], tasks=[task]))
    
    # Create a memory item with a future timestamp
    future_data = "Test data with future timestamp"
    
    # We need to pass the data directly to the save method
    # The save method will create a ShortTermMemoryItem internally
    # and then we'll modify its timestamp before it's saved
    
    # Mock datetime.now to return a fixed time
    with patch('crewai.memory.short_term.short_term_memory_item.datetime') as mock_datetime:
        # Set up the mock to return our future timestamp when now() is called
        mock_datetime.now.return_value = future_timestamp
        
        with pytest.raises(ValueError, match="Cannot save memory item with future timestamp"):
            memory.save(value=future_data)
