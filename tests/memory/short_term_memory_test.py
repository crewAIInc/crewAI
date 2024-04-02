import pytest

from crewai.memory.short_term.short_term_memory import ShortTermMemory
from crewai.memory.short_term.short_term_memory_item import ShortTermMemoryItem


@pytest.fixture
def short_term_memory():
    """Fixture to create a ShortTermMemory instance"""
    return ShortTermMemory()


def test_save_and_search(short_term_memory):
    memory = ShortTermMemoryItem(
        data="""test value test value test value test value test value test value
        test value test value test value test value test value test value
        test value test value test value test value test value test value""",
        agent="test_agent",
        metadata={"task": "test_task"},
    )
    short_term_memory.save(memory)
    find = short_term_memory.search("test value", score_threshold=0.01)[0]
    assert find["context"] == memory.data, "Data value mismatch."
    assert find["metadata"]["agent"] == "test_agent", "Agent value mismatch."
