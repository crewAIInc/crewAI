import inspect
import threading
from collections import defaultdict
from typing import Any
from unittest.mock import ANY

import pytest

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.memory_events import (
    MemoryQueryCompletedEvent,
    MemoryQueryStartedEvent,
    MemorySaveCompletedEvent,
    MemorySaveStartedEvent,
)
from crewai.memory.long_term.long_term_memory import LongTermMemory
from crewai.memory.long_term.long_term_memory_item import LongTermMemoryItem
from crewai.memory.memory import Memory


@pytest.fixture
def long_term_memory():
    """Fixture to create a LongTermMemory instance"""
    return LongTermMemory()


def test_long_term_memory_save_events(long_term_memory):
    events = defaultdict(list)
    condition = threading.Condition()

    @crewai_event_bus.on(MemorySaveStartedEvent)
    def on_save_started(source, event):
        with condition:
            events["MemorySaveStartedEvent"].append(event)
            condition.notify()

    @crewai_event_bus.on(MemorySaveCompletedEvent)
    def on_save_completed(source, event):
        with condition:
            events["MemorySaveCompletedEvent"].append(event)
            condition.notify()

    memory = LongTermMemoryItem(
        agent="test_agent",
        task="test_task",
        expected_output="test_output",
        datetime="test_datetime",
        quality=0.5,
        metadata={"task": "test_task", "quality": 0.5},
    )
    long_term_memory.save(memory)

    with condition:
        success = condition.wait_for(
            lambda: len(events["MemorySaveStartedEvent"]) >= 1
            and len(events["MemorySaveCompletedEvent"]) >= 1,
            timeout=5,
        )
    assert success, "Timeout waiting for save events"
    assert len(events["MemorySaveStartedEvent"]) == 1
    assert len(events["MemorySaveCompletedEvent"]) == 1
    assert len(events["MemorySaveFailedEvent"]) == 0

    assert dict(events["MemorySaveStartedEvent"][0]) == {
        "timestamp": ANY,
        "type": "memory_save_started",
        "source_fingerprint": None,
        "source_type": "long_term_memory",
        "fingerprint_metadata": None,
        "task_id": None,
        "task_name": None,
        "from_task": None,
        "from_agent": None,
        "agent_role": "test_agent",
        "agent_id": None,
        "value": "test_task",
        "metadata": {"task": "test_task", "quality": 0.5},
    }
    assert dict(events["MemorySaveCompletedEvent"][0]) == {
        "timestamp": ANY,
        "type": "memory_save_completed",
        "source_fingerprint": None,
        "source_type": "long_term_memory",
        "fingerprint_metadata": None,
        "task_id": None,
        "task_name": None,
        "from_task": None,
        "from_agent": None,
        "agent_role": "test_agent",
        "agent_id": None,
        "value": "test_task",
        "metadata": {
            "task": "test_task",
            "quality": 0.5,
            "agent": "test_agent",
            "expected_output": "test_output",
        },
        "save_time_ms": ANY,
    }


def test_long_term_memory_search_events(long_term_memory):
    events = defaultdict(list)
    condition = threading.Condition()

    @crewai_event_bus.on(MemoryQueryStartedEvent)
    def on_search_started(source, event):
        with condition:
            events["MemoryQueryStartedEvent"].append(event)
            condition.notify()

    @crewai_event_bus.on(MemoryQueryCompletedEvent)
    def on_search_completed(source, event):
        with condition:
            events["MemoryQueryCompletedEvent"].append(event)
            condition.notify()

    test_query = "test query"

    long_term_memory.search(test_query, limit=5)

    with condition:
        success = condition.wait_for(
            lambda: len(events["MemoryQueryStartedEvent"]) >= 1
            and len(events["MemoryQueryCompletedEvent"]) >= 1,
            timeout=5,
        )
    assert success, "Timeout waiting for search events"
    assert len(events["MemoryQueryStartedEvent"]) == 1
    assert len(events["MemoryQueryCompletedEvent"]) == 1
    assert len(events["MemoryQueryFailedEvent"]) == 0

    assert dict(events["MemoryQueryStartedEvent"][0]) == {
        "timestamp": ANY,
        "type": "memory_query_started",
        "source_fingerprint": None,
        "source_type": "long_term_memory",
        "fingerprint_metadata": None,
        "task_id": None,
        "task_name": None,
        "from_task": None,
        "from_agent": None,
        "agent_role": None,
        "agent_id": None,
        "query": "test query",
        "limit": 5,
        "score_threshold": None,
    }

    assert dict(events["MemoryQueryCompletedEvent"][0]) == {
        "timestamp": ANY,
        "type": "memory_query_completed",
        "source_fingerprint": None,
        "source_type": "long_term_memory",
        "fingerprint_metadata": None,
        "task_id": None,
        "task_name": None,
        "from_task": None,
        "from_agent": None,
        "agent_role": None,
        "agent_id": None,
        "query": "test query",
        "results": None,
        "limit": 5,
        "score_threshold": None,
        "query_time_ms": ANY,
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
    find = long_term_memory.search("test_task", limit=5)[0]
    assert find["score"] == 0.5
    assert find["datetime"] == "test_datetime"
    assert find["metadata"]["agent"] == "test_agent"
    assert find["metadata"]["quality"] == 0.5
    assert find["metadata"]["task"] == "test_task"
    assert find["metadata"]["expected_output"] == "test_output"


class TestLongTermMemoryTypeSignatureCompatibility:
    """Tests to verify LongTermMemory method signatures are compatible with Memory base class.

    These tests ensure that the Liskov Substitution Principle is maintained and that
    LongTermMemory can be used polymorphically wherever Memory is expected.
    """

    def test_save_signature_has_value_parameter(self):
        """Test that save() uses 'value' parameter name matching Memory base class."""
        sig = inspect.signature(LongTermMemory.save)
        params = list(sig.parameters.keys())
        assert "value" in params, "save() should have 'value' parameter for LSP compliance"
        assert "metadata" in params, "save() should have 'metadata' parameter for LSP compliance"

    def test_save_signature_has_metadata_with_default(self):
        """Test that save() has metadata parameter with default value."""
        sig = inspect.signature(LongTermMemory.save)
        metadata_param = sig.parameters.get("metadata")
        assert metadata_param is not None, "save() should have 'metadata' parameter"
        assert metadata_param.default is None, "metadata should default to None"

    def test_search_signature_has_query_parameter(self):
        """Test that search() uses 'query' parameter name matching Memory base class."""
        sig = inspect.signature(LongTermMemory.search)
        params = list(sig.parameters.keys())
        assert "query" in params, "search() should have 'query' parameter for LSP compliance"
        assert "limit" in params, "search() should have 'limit' parameter for LSP compliance"
        assert "score_threshold" in params, "search() should have 'score_threshold' parameter for LSP compliance"

    def test_search_signature_has_score_threshold_with_default(self):
        """Test that search() has score_threshold parameter with default value."""
        sig = inspect.signature(LongTermMemory.search)
        score_threshold_param = sig.parameters.get("score_threshold")
        assert score_threshold_param is not None, "search() should have 'score_threshold' parameter"
        assert score_threshold_param.default == 0.6, "score_threshold should default to 0.6"

    def test_asave_signature_has_value_parameter(self):
        """Test that asave() uses 'value' parameter name matching Memory base class."""
        sig = inspect.signature(LongTermMemory.asave)
        params = list(sig.parameters.keys())
        assert "value" in params, "asave() should have 'value' parameter for LSP compliance"
        assert "metadata" in params, "asave() should have 'metadata' parameter for LSP compliance"

    def test_asearch_signature_has_query_parameter(self):
        """Test that asearch() uses 'query' parameter name matching Memory base class."""
        sig = inspect.signature(LongTermMemory.asearch)
        params = list(sig.parameters.keys())
        assert "query" in params, "asearch() should have 'query' parameter for LSP compliance"
        assert "limit" in params, "asearch() should have 'limit' parameter for LSP compliance"
        assert "score_threshold" in params, "asearch() should have 'score_threshold' parameter for LSP compliance"

    def test_long_term_memory_is_subclass_of_memory(self):
        """Test that LongTermMemory is a proper subclass of Memory."""
        assert issubclass(LongTermMemory, Memory), "LongTermMemory should be a subclass of Memory"

    def test_save_with_metadata_parameter(self, long_term_memory):
        """Test that save() can be called with the metadata parameter (even if unused)."""
        memory_item = LongTermMemoryItem(
            agent="test_agent",
            task="test_task_with_metadata",
            expected_output="test_output",
            datetime="test_datetime",
            quality=0.8,
            metadata={"task": "test_task_with_metadata", "quality": 0.8},
        )
        long_term_memory.save(value=memory_item, metadata={"extra": "data"})
        results = long_term_memory.search(query="test_task_with_metadata", limit=1)
        assert len(results) > 0
        assert results[0]["metadata"]["agent"] == "test_agent"

    def test_search_with_score_threshold_parameter(self, long_term_memory):
        """Test that search() can be called with the score_threshold parameter."""
        memory_item = LongTermMemoryItem(
            agent="test_agent",
            task="test_task_score_threshold",
            expected_output="test_output",
            datetime="test_datetime",
            quality=0.9,
            metadata={"task": "test_task_score_threshold", "quality": 0.9},
        )
        long_term_memory.save(value=memory_item)
        results = long_term_memory.search(
            query="test_task_score_threshold",
            limit=5,
            score_threshold=0.5,
        )
        assert isinstance(results, list)

    @pytest.fixture
    def long_term_memory(self):
        """Fixture to create a LongTermMemory instance for this test class."""
        return LongTermMemory()
