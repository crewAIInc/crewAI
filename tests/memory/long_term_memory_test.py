import pytest
from unittest.mock import ANY
from collections import defaultdict
from crewai.utilities.events import crewai_event_bus
from crewai.memory.long_term.long_term_memory import LongTermMemory
from crewai.memory.long_term.long_term_memory_item import LongTermMemoryItem
from crewai.utilities.events.memory_events import (
    MemorySaveStartedEvent,
    MemorySaveCompletedEvent,
    MemoryQueryStartedEvent,
    MemoryQueryCompletedEvent,
)

@pytest.fixture
def long_term_memory():
    """Fixture to create a LongTermMemory instance"""
    return LongTermMemory()


def test_long_term_memory_save_events(long_term_memory):
    events = defaultdict(list)

    with crewai_event_bus.scoped_handlers():
        @crewai_event_bus.on(MemorySaveStartedEvent)
        def on_save_started(source, event):
            events["MemorySaveStartedEvent"].append(event)

        @crewai_event_bus.on(MemorySaveCompletedEvent)
        def on_save_completed(source, event):
            events["MemorySaveCompletedEvent"].append(event)

        memory = LongTermMemoryItem(
            agent="test_agent",
            task="test_task",
            expected_output="test_output",
            datetime="test_datetime",
            quality=0.5,
            metadata={"task": "test_task", "quality": 0.5},
        )
        long_term_memory.save(memory)

    assert len(events["MemorySaveStartedEvent"]) == 1
    assert len(events["MemorySaveCompletedEvent"]) == 1
    assert len(events["MemorySaveFailedEvent"]) == 0

    assert dict(events["MemorySaveStartedEvent"][0]) == {
        "timestamp": ANY,
        "type": "memory_save_started",
        "source_fingerprint": None,
        "source_type": "long_term_memory",
        "fingerprint_metadata": None,
        "value": "test_task",
        "metadata": {"task": "test_task", "quality": 0.5},
        "agent_role": "test_agent",
    }
    assert dict(events["MemorySaveCompletedEvent"][0]) == {
        "timestamp": ANY,
        "type": "memory_save_completed",
        "source_fingerprint": None,
        "source_type": "long_term_memory",
        "fingerprint_metadata": None,
        "value": "test_task",
        "metadata": {"task": "test_task", "quality": 0.5, "agent": "test_agent", "expected_output": "test_output"},
        "agent_role": "test_agent",
        "save_time_ms": ANY,
    }


def test_long_term_memory_search_events(long_term_memory):
    events = defaultdict(list)

    with crewai_event_bus.scoped_handlers():
        @crewai_event_bus.on(MemoryQueryStartedEvent)
        def on_search_started(source, event):
            events["MemoryQueryStartedEvent"].append(event)

        @crewai_event_bus.on(MemoryQueryCompletedEvent)
        def on_search_completed(source, event):
            events["MemoryQueryCompletedEvent"].append(event)

        test_query = "test query"

        long_term_memory.search(
            test_query,
            latest_n=5
        )

    assert len(events["MemoryQueryStartedEvent"]) == 1
    assert len(events["MemoryQueryCompletedEvent"]) == 1
    assert len(events["MemoryQueryFailedEvent"]) == 0

    assert dict(events["MemoryQueryStartedEvent"][0]) == {
        'timestamp': ANY,
        'type': 'memory_query_started',
        'source_fingerprint': None,
        'source_type': 'long_term_memory',
        'fingerprint_metadata': None,
        'query': 'test query',
        'limit': 5,
        'score_threshold': None
    }

    assert dict(events["MemoryQueryCompletedEvent"][0]) == {
        'timestamp': ANY,
        'type': 'memory_query_completed',
        'source_fingerprint': None,
        'source_type': 'long_term_memory',
        'fingerprint_metadata': None,
        'query': 'test query',
        'results': None,
        'limit': 5,
        'score_threshold': None,
        'query_time_ms': ANY
    }


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
