"""Regression tests: listener in-progress flags must reset on failure paths.

TraceCollectionListener and EventListener are process-wide singletons that
guard console formatter pairings behind boolean flags. The failed paths must
clear the flags, or every later save/retrieval in the process loses its
"started" display: one failed memory save used to suppress all subsequent
save displays forever, and a failed retrieval had no handler at all.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from crewai.events.event_bus import crewai_event_bus
from crewai.events.listeners.tracing.trace_listener import TraceCollectionListener
from crewai.events.types.memory_events import (
    MemoryRetrievalCompletedEvent,
    MemoryRetrievalFailedEvent,
    MemoryRetrievalStartedEvent,
    MemorySaveCompletedEvent,
    MemorySaveFailedEvent,
    MemorySaveStartedEvent,
)


def _emit(event) -> None:
    future = crewai_event_bus.emit(None, event)
    if future:
        future.result(timeout=5)


def _wait_for(condition, timeout: float = 5.0) -> bool:
    """Handler dispatch can be scheduled; wait for the assertion to hold."""
    import time as _time

    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        if condition():
            return True
        _time.sleep(0.05)
    return False


@pytest.fixture
def listener_with_mock_formatter():
    """Register handlers with the tracing gate forced on; re-setup per test
    because cleanup_event_handlers wipes the bus between tests (the singleton's
    _listeners_setup flag alone would make later setups no-ops)."""
    with (
        patch.dict(os.environ, {"CREWAI_TRACING_ENABLED": "true"}),
        patch("crewai.events.listeners.tracing.trace_listener.should_enable_tracing", return_value=True),
    ):
        listener = TraceCollectionListener()
        listener._listeners_setup = False
        listener.setup_listeners(crewai_event_bus)
        formatter = MagicMock()
        listener.formatter = formatter
        listener.memory_save_in_progress = False
        listener.memory_retrieval_in_progress = False
        try:
            yield listener, formatter
        finally:
            TraceCollectionListener._listeners_setup = False


def test_memory_save_started_fires_again_after_failed_save(listener_with_mock_formatter) -> None:
    listener, formatter = listener_with_mock_formatter

    _emit(MemorySaveStartedEvent(value="m1", source_type="unified_memory"))
    _emit(MemorySaveFailedEvent(value="m1", error="boom", source_type="unified_memory"))
    assert listener.memory_save_in_progress is False, "failed save must clear the in-progress flag"

    _emit(MemorySaveStartedEvent(value="m2", source_type="unified_memory"))
    _emit(MemorySaveCompletedEvent(value="m2", save_time_ms=1.0, source_type="unified_memory"))

    assert formatter.handle_memory_save_started.call_count == 2, (
        "a failed save must not swallow later saves' started display"
    )


def test_memory_retrieval_started_fires_again_after_failed_retrieval(listener_with_mock_formatter) -> None:
    listener, formatter = listener_with_mock_formatter

    _emit(MemoryRetrievalStartedEvent())
    _emit(MemoryRetrievalFailedEvent(error="boom"))
    assert listener.memory_retrieval_in_progress is False, (
        "failed retrieval must clear the in-progress flag"
    )

    _emit(MemoryRetrievalStartedEvent())
    _emit(MemoryRetrievalCompletedEvent(memory_content="ok", retrieval_time_ms=1.0))

    assert formatter.handle_memory_retrieval_started.call_count == 2, (
        "a failed retrieval must not swallow later retrievals' started display"
    )


def test_empty_knowledge_query_still_emits_completed_event() -> None:
    """The sync retrieval path must not orphan the Started event.

    When the query rewrite returns None the retrieval still started; without a
    matching Completed event, the process-wide listener's in-progress flag
    sticks and swallows every later retrieval's display.
    """
    from crewai.agent.utils import handle_knowledge_retrieval
    from crewai.events.types.knowledge_events import (
        KnowledgeRetrievalCompletedEvent,
        KnowledgeRetrievalStartedEvent,
    )

    agent = MagicMock()
    agent.crew = "none"
    agent.knowledge = object()  # truthy: the retrieval starts
    agent._get_knowledge_search_query.return_value = None

    emitted: list[str] = []
    on_started = lambda source, event: emitted.append("started")
    on_completed = lambda source, event: emitted.append("completed")
    crewai_event_bus.register_handler(KnowledgeRetrievalStartedEvent, on_started)
    crewai_event_bus.register_handler(KnowledgeRetrievalCompletedEvent, on_completed)
    try:
        result = handle_knowledge_retrieval(
            agent, MagicMock(), "prompt", {}, lambda *a, **k: [], lambda *a, **k: []
        )
    finally:
        crewai_event_bus.off(KnowledgeRetrievalStartedEvent, on_started)
        crewai_event_bus.off(KnowledgeRetrievalCompletedEvent, on_completed)

    assert result == "prompt"
    assert _wait_for(lambda: len(emitted) >= 2), f"events never arrived: {emitted!r}"
    assert emitted == ["started", "completed"], (
        "an empty query must still end the retrieval so listeners release state"
    )


def test_empty_knowledge_query_still_emits_completed_event_async() -> None:
    """The async retrieval twin must not orphan the Started event either."""
    import asyncio

    from crewai.agent.utils import ahandle_knowledge_retrieval
    from crewai.events.types.knowledge_events import (
        KnowledgeRetrievalCompletedEvent,
        KnowledgeRetrievalStartedEvent,
    )

    agent = MagicMock()
    agent.crew = "none"
    agent.knowledge = object()
    agent._get_knowledge_search_query.return_value = None

    emitted: list[str] = []
    on_started = lambda source, event: emitted.append("started")
    on_completed = lambda source, event: emitted.append("completed")
    crewai_event_bus.register_handler(KnowledgeRetrievalStartedEvent, on_started)
    crewai_event_bus.register_handler(KnowledgeRetrievalCompletedEvent, on_completed)
    try:
        result = asyncio.run(ahandle_knowledge_retrieval(agent, MagicMock(), "prompt", {}))
    finally:
        crewai_event_bus.off(KnowledgeRetrievalStartedEvent, on_started)
        crewai_event_bus.off(KnowledgeRetrievalCompletedEvent, on_completed)

    assert result == "prompt"
    assert _wait_for(lambda: len(emitted) >= 2), f"events never arrived: {emitted!r}"
    assert emitted == ["started", "completed"]
