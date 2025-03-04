from typing import Any, Dict, List

import pytest

from crewai.agent import Agent
from crewai.crew import Crew
from crewai.memory.entity.entity_memory import EntityMemory
from crewai.memory.long_term.long_term_memory import LongTermMemory
from crewai.memory.short_term.short_term_memory import ShortTermMemory
from crewai.memory.storage.interface import Storage, SearchResult
from crewai.memory.user.user_memory import UserMemory


class CustomStorage(Storage):
    """Custom storage implementation for testing."""

    def __init__(self):
        self.data = []

    def save(self, value: Any, metadata: Dict[str, Any]) -> None:
        self.data.append({"value": value, "metadata": metadata})

    def search(
        self, query: str, limit: int = 3, score_threshold: float = 0.35
    ) -> List[SearchResult]:
        return [{"context": item["value"], "metadata": item["metadata"], "score": 0.9} for item in self.data]

    def reset(self) -> None:
        self.data = []


def test_custom_storage_with_short_term_memory():
    """Test that custom storage works with short term memory."""
    custom_storage = CustomStorage()
    memory = ShortTermMemory(storage=custom_storage)
    
    memory.save("test value", {"key": "value"})
    results = memory.search("test")
    
    assert len(results) > 0
    assert results[0]["context"] == "test value"
    assert results[0]["metadata"]["key"] == "value"


def test_custom_storage_with_long_term_memory():
    """Test that custom storage works with long term memory."""
    custom_storage = CustomStorage()
    memory = LongTermMemory(storage=custom_storage)
    
    memory.save("test value", {"key": "value"})
    results = memory.search("test")
    
    assert len(results) > 0
    assert results[0]["context"] == "test value"
    assert results[0]["metadata"]["key"] == "value"


def test_custom_storage_with_entity_memory():
    """Test that custom storage works with entity memory."""
    custom_storage = CustomStorage()
    memory = EntityMemory(storage=custom_storage)
    
    memory.save("test value", {"key": "value"})
    results = memory.search("test")
    
    assert len(results) > 0
    assert results[0]["context"] == "test value"
    assert results[0]["metadata"]["key"] == "value"


def test_custom_storage_with_user_memory():
    """Test that custom storage works with user memory."""
    custom_storage = CustomStorage()
    memory = UserMemory(storage=custom_storage)
    
    memory.save("test value", {"key": "value"})
    results = memory.search("test")
    
    assert len(results) > 0
    # UserMemory prepends "Remember the details about the user: " to the value
    assert "test value" in results[0]["context"]
    assert results[0]["metadata"]["key"] == "value"


def test_custom_storage_with_crew():
    """Test that custom storage works with crew."""
    short_term_storage = CustomStorage()
    long_term_storage = CustomStorage()
    entity_storage = CustomStorage()
    user_storage = CustomStorage()
    
    # Create memory instances with custom storage
    short_term_memory = ShortTermMemory(storage=short_term_storage)
    long_term_memory = LongTermMemory(storage=long_term_storage)
    entity_memory = EntityMemory(storage=entity_storage)
    user_memory = UserMemory(storage=user_storage)
    
    # Create a crew with custom memory instances
    crew = Crew(
        agents=[Agent(role="test", goal="test", backstory="test")],
        memory=True,
        short_term_memory=short_term_memory,
        long_term_memory=long_term_memory,
        entity_memory=entity_memory,
        memory_config={"user_memory": user_memory},
    )
    
    # Test that the crew has the custom memory instances
    assert crew._short_term_memory.storage == short_term_storage
    assert crew._long_term_memory.storage == long_term_storage
    assert crew._entity_memory.storage == entity_storage
    assert crew._user_memory.storage == user_storage


def test_custom_storage_with_memory_config():
    """Test that custom storage works with memory_config."""
    short_term_storage = CustomStorage()
    long_term_memory = LongTermMemory(storage=CustomStorage())
    entity_memory = EntityMemory(storage=CustomStorage())
    user_memory = UserMemory(storage=CustomStorage())
    
    # Create a crew with custom storage in memory_config
    crew = Crew(
        agents=[Agent(role="test", goal="test", backstory="test")],
        memory=True,
        short_term_memory=ShortTermMemory(storage=short_term_storage),
        long_term_memory=long_term_memory,
        entity_memory=entity_memory,
        memory_config={
            "user_memory": user_memory
        },
    )
    
    # Test that the crew has the custom storage instances
    assert crew._short_term_memory.storage == short_term_storage
    assert crew._long_term_memory == long_term_memory
    assert crew._entity_memory == entity_memory
    assert crew._user_memory == user_memory


def test_custom_storage_error_handling():
    """Test error handling with custom storage."""
    # Test exception propagation
    class ErrorStorage(Storage):
        """Storage implementation that raises exceptions."""
        def __init__(self):
            self.data = []
            
        def save(self, value: Any, metadata: Dict[str, Any]) -> None:
            raise ValueError("Save error")
            
        def search(
            self, query: str, limit: int = 3, score_threshold: float = 0.35
        ) -> List[SearchResult]:
            raise ValueError("Search error")
            
        def reset(self) -> None:
            raise ValueError("Reset error")
            
    storage = ErrorStorage()
    memory = ShortTermMemory(storage=storage)
    
    with pytest.raises(ValueError, match="Save error"):
        memory.save("test", {})
        
    with pytest.raises(ValueError, match="Search error"):
        memory.search("test")
        
    with pytest.raises(Exception, match="An error occurred while resetting the short-term memory: Reset error"):
        memory.reset()


def test_custom_storage_edge_cases():
    """Test edge cases with custom storage."""
    class EdgeCaseStorage(Storage):
        """Storage implementation for testing edge cases."""
        def __init__(self):
            self.data = []
            
        def save(self, value: Any, metadata: Dict[str, Any]) -> None:
            self.data.append({"value": value, "metadata": metadata})
            
        def search(
            self, query: str, limit: int = 3, score_threshold: float = 0.35
        ) -> List[SearchResult]:
            return [{"context": item["value"], "metadata": item["metadata"], "score": 0.5} for item in self.data]
            
        def reset(self) -> None:
            self.data = []
            
    storage = EdgeCaseStorage()
    memory = ShortTermMemory(storage=storage)
    
    # Test empty query
    memory.save("test value", {"key": "value"})
    results = memory.search("")
    assert len(results) > 0
    
    # Test very large metadata
    large_metadata = {"key" + str(i): "value" * 100 for i in range(100)}
    memory.save("test value", large_metadata)
    results = memory.search("test")
    assert len(results) > 0
    assert results[1]["metadata"] == large_metadata
    
    # Test unicode and special characters
    unicode_value = "测试值 with special chars: !@#$%^&*()"
    memory.save(unicode_value, {"key": "value"})
    results = memory.search("测试")
    assert len(results) > 0
    assert unicode_value in results[2]["context"]
