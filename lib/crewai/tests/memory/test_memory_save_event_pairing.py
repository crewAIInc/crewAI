"""Regression tests for memory-save failure event pairing.

``Memory._on_save_done`` used to emit ``MemorySaveFailedEvent`` from the
save pool's done callback, which runs in the pool thread's own context where
the event-bus scope stack is empty. A failing save therefore produced an
unpaired ending event (bus warning, dangling trace span) — and for the
synchronous ``remember()`` a *duplicate* on top of the caller's own paired
failed event. The background ``remember_many()`` path additionally left its
started event unterminated, because its failed event only ever came from
that orphan callback.
"""

from typing import Any

import pytest

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.memory_events import (
    MemorySaveCompletedEvent,
    MemorySaveFailedEvent,
    MemorySaveStartedEvent,
)
from crewai.memory.unified_memory import Memory


class MemorySaveEventRecorder:
    def __init__(self):
        self.started = []
        self.completed = []
        self.failed = []

    def register(self):
        crewai_event_bus.register_handler(MemorySaveStartedEvent, self._on_started)
        crewai_event_bus.register_handler(MemorySaveCompletedEvent, self._on_completed)
        crewai_event_bus.register_handler(MemorySaveFailedEvent, self._on_failed)

    def unregister(self):
        crewai_event_bus.off(MemorySaveStartedEvent, self._on_started)
        crewai_event_bus.off(MemorySaveCompletedEvent, self._on_completed)
        crewai_event_bus.off(MemorySaveFailedEvent, self._on_failed)

    def _on_started(self, source, event):
        self.started.append(event)

    def _on_completed(self, source, event):
        self.completed.append(event)

    def _on_failed(self, source, event):
        self.failed.append(event)


def make_failing_memory() -> Memory:
    """A Memory whose encoding pipeline always fails."""

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("embedder exploded")

    mem = Memory()
    mem._encode_batch = boom  # type: ignore[method-assign]
    return mem


def test_sync_remember_failure_emits_one_paired_failed():
    """A failing remember() must emit exactly one failed event, paired."""
    mem = make_failing_memory()
    recorder = MemorySaveEventRecorder()
    recorder.register()
    try:
        with pytest.raises(RuntimeError, match="embedder exploded"):
            mem.remember("sync memory")
        crewai_event_bus.flush()
    finally:
        recorder.unregister()

    assert len(recorder.started) == 1
    assert len(recorder.failed) == 1, (
        "a single save failure must not produce a duplicate failed event "
        "from the pool callback"
    )
    assert recorder.failed[0].started_event_id == recorder.started[0].event_id


def test_background_remember_many_failure_pairs_events():
    """A failing background save must terminate its own started scope."""
    mem = make_failing_memory()
    recorder = MemorySaveEventRecorder()
    recorder.register()
    try:
        mem.remember_many(["background memory"])
        mem.drain_writes()
        crewai_event_bus.flush()
    finally:
        recorder.unregister()

    assert len(recorder.started) == 1
    assert len(recorder.completed) == 0
    assert len(recorder.failed) == 1, (
        "the background save must emit its own failed event, not rely on "
        "the pool callback whose context has an empty scope stack"
    )
    assert recorder.failed[0].started_event_id == recorder.started[0].event_id
