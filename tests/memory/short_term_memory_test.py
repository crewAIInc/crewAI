from unittest.mock import patch, ANY
from collections import defaultdict
import pytest

from crewai.agent import Agent
from crewai.crew import Crew
from crewai.memory.short_term.short_term_memory import ShortTermMemory
from crewai.memory.short_term.short_term_memory_item import ShortTermMemoryItem
from crewai.task import Task
from crewai.utilities.events import crewai_event_bus
from crewai.utilities.events.memory_events import (
    MemorySaveStartedEvent,
    MemorySaveCompletedEvent,
    MemoryQueryStartedEvent,
    MemoryQueryCompletedEvent,
)


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


def test_short_term_memory_search_events(short_term_memory):
    events = defaultdict(list)

    with crewai_event_bus.scoped_handlers():
        @crewai_event_bus.on(MemoryQueryStartedEvent)
        def on_search_started(source, event):
            events["MemoryQueryStartedEvent"].append(event)

        @crewai_event_bus.on(MemoryQueryCompletedEvent)
        def on_search_completed(source, event):
            events["MemoryQueryCompletedEvent"].append(event)

        # Call the save method
        short_term_memory.search(
            query="test value",
            limit=3,
            score_threshold=0.35,
        )

    assert len(events["MemoryQueryStartedEvent"]) == 1
    assert len(events["MemoryQueryCompletedEvent"]) == 1
    assert len(events["MemoryQueryFailedEvent"]) == 0

    assert dict(events["MemoryQueryStartedEvent"][0]) == {
        'timestamp': ANY,
        'type': 'memory_query_started',
        'source_fingerprint': None,
        'source_type': 'short_term_memory',
        'fingerprint_metadata': None,
        'query': 'test value',
        'limit': 3,
        'score_threshold': 0.35
    }

    assert dict(events["MemoryQueryCompletedEvent"][0]) == {
        'timestamp': ANY,
        'type': 'memory_query_completed',
        'source_fingerprint': None,
        'source_type': 'short_term_memory',
        'fingerprint_metadata': None,
        'query': 'test value',
        'results': [],
        'limit': 3,
        'score_threshold': 0.35,
        'query_time_ms': ANY
    }


def test_short_term_memory_save_events(short_term_memory):
    events = defaultdict(list)
    with crewai_event_bus.scoped_handlers():
        @crewai_event_bus.on(MemorySaveStartedEvent)
        def on_save_started(source, event):
            events["MemorySaveStartedEvent"].append(event)

        @crewai_event_bus.on(MemorySaveCompletedEvent)
        def on_save_completed(source, event):
            events["MemorySaveCompletedEvent"].append(event)

        short_term_memory.save(
            value="test value",
            metadata={"task": "test_task"},
            agent="test_agent",
        )

    assert len(events["MemorySaveStartedEvent"]) == 1
    assert len(events["MemorySaveCompletedEvent"]) == 1
    assert len(events["MemorySaveFailedEvent"]) == 0

    assert dict(events["MemorySaveStartedEvent"][0]) == {
        'timestamp': ANY,
        'type': 'memory_save_started',
        'source_fingerprint': None,
        'source_type': 'short_term_memory',
        'fingerprint_metadata': None,
        'value': 'test value',
        'metadata': {'task': 'test_task'},
        'agent_role': "test_agent"
    }

    assert dict(events["MemorySaveCompletedEvent"][0]) == {
        'timestamp': ANY,
        'type': 'memory_save_completed',
        'source_fingerprint': None,
        'source_type': 'short_term_memory',
        'fingerprint_metadata': None,
        'value': 'test value',
        'metadata': {'task': 'test_task', 'agent': 'test_agent'},
        'agent_role': "test_agent",
        'save_time_ms': ANY
    }

def test_save_and_search(short_term_memory):
    memory = ShortTermMemoryItem(
        data="""test value test value test value test value test value test value
        test value test value test value test value test value test value
        test value test value test value test value test value test value""",
        agent="test_agent",
        metadata={"task": "test_task"},
    )

    with patch.object(ShortTermMemory, "save") as mock_save:
        short_term_memory.save(
            value=memory.data,
            metadata=memory.metadata,
            agent=memory.agent,
        )

        mock_save.assert_called_once_with(
            value=memory.data,
            metadata=memory.metadata,
            agent=memory.agent,
        )

    expected_result = [
        {
            "context": memory.data,
            "metadata": {"agent": "test_agent"},
            "score": 0.95,
        }
    ]
    with patch.object(ShortTermMemory, "search", return_value=expected_result):
        find = short_term_memory.search("test value", score_threshold=0.01)[0]
        assert find["context"] == memory.data, "Data value mismatch."
        assert find["metadata"]["agent"] == "test_agent", "Agent value mismatch."
