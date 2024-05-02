import pytest

from crewai.memory.long_term.long_term_memory import LongTermMemory
from crewai.memory.long_term.long_term_memory_item import LongTermMemoryItem


@pytest.fixture
def long_term_memory():
    """Fixture to create a LongTermMemory instance"""
    return LongTermMemory()


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
    find = long_term_memory.search("test_task", latest_n=5)[0]
    assert find["score"] == 0.5
    assert find["datetime"] == "test_datetime"
    assert find["metadata"]["agent"] == "test_agent"
    assert find["metadata"]["quality"] == 0.5
    assert find["metadata"]["task"] == "test_task"
    assert find["metadata"]["expected_output"] == "test_output"
