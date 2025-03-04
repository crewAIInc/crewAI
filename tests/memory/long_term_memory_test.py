import pytest

from crewai.memory.long_term.long_term_memory import LongTermMemory
from crewai.memory.long_term.long_term_memory_item import LongTermMemoryItem


@pytest.fixture
def long_term_memory():
    """Fixture to create a LongTermMemory instance"""
    # Create a mock storage for testing
    from crewai.memory.storage.interface import Storage
    
    class MockStorage(Storage):
        def __init__(self):
            self.data = []
            
        def save(self, value, metadata):
            self.data.append({"value": value, "metadata": metadata})
            
        def search(self, query, limit=3, score_threshold=0.35):
            return [
                {
                    "context": item["value"],
                    "metadata": item["metadata"],
                    "score": 0.5,
                    "datetime": item["metadata"].get("datetime", "test_datetime")
                }
                for item in self.data
            ]
            
        def reset(self):
            self.data = []
            
    return LongTermMemory(storage=MockStorage())


def test_save_and_search(long_term_memory):
    memory = LongTermMemoryItem(
        agent="test_agent",
        task="test_task",
        expected_output="test_output",
        datetime="test_datetime",
        quality=0.5,
        metadata={"task": "test_task", "quality": 0.5},
    )
    long_term_memory.save(memory)
    find = long_term_memory.search(query="test_task", limit=5)[0]
    assert find["score"] == 0.5
    assert find["datetime"] == "test_datetime"
    assert find["metadata"]["agent"] == "test_agent"
    assert find["metadata"]["quality"] == 0.5
    assert find["metadata"]["task"] == "test_task"
    assert find["metadata"]["expected_output"] == "test_output"
