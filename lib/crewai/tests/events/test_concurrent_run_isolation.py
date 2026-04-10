"""Tests for event bus concurrent run/stream isolation.

Verifies that events emitted within a ``run_scope()`` are tagged with the
correct ``run_id`` and that stream handlers only receive events belonging
to their own run, preventing cross-run chunk contamination (GitHub #5376).
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from crewai.events.base_events import BaseEvent
from crewai.events.event_bus import (
    CrewAIEventsBus,
    crewai_event_bus,
    get_current_run_id,
    run_scope,
    set_current_run_id,
)
from crewai.events.types.llm_events import LLMStreamChunkEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _SimpleEvent(BaseEvent):
    """Minimal concrete event for testing."""

    type: str = "test_simple_event"


def _make_stream_chunk(
    chunk: str,
    call_id: str = "test-call",
    run_id: str | None = None,
) -> LLMStreamChunkEvent:
    """Create a minimal LLMStreamChunkEvent for testing."""
    evt = LLMStreamChunkEvent(chunk=chunk, call_id=call_id)
    if run_id is not None:
        evt.run_id = run_id
    return evt


# ---------------------------------------------------------------------------
# 1. run_scope basics
# ---------------------------------------------------------------------------

class TestRunScope:
    """Tests for the run_scope context manager."""

    def test_run_scope_sets_and_resets_run_id(self) -> None:
        """run_scope sets run_id inside and clears it outside."""
        assert get_current_run_id() is None
        with run_scope() as rid:
            assert rid is not None
            assert get_current_run_id() == rid
        assert get_current_run_id() is None

    def test_run_scope_with_explicit_id(self) -> None:
        """run_scope accepts an explicit run_id."""
        with run_scope("my-custom-id") as rid:
            assert rid == "my-custom-id"
            assert get_current_run_id() == "my-custom-id"
        assert get_current_run_id() is None

    def test_nested_run_scopes(self) -> None:
        """Nested run_scopes correctly stack and unwind."""
        with run_scope("outer") as outer_id:
            assert get_current_run_id() == "outer"
            with run_scope("inner") as inner_id:
                assert get_current_run_id() == "inner"
            assert get_current_run_id() == "outer"
        assert get_current_run_id() is None

    def test_set_current_run_id(self) -> None:
        """set_current_run_id can be used directly."""
        set_current_run_id("explicit")
        assert get_current_run_id() == "explicit"
        set_current_run_id(None)
        assert get_current_run_id() is None


# ---------------------------------------------------------------------------
# 2. Event stamping
# ---------------------------------------------------------------------------

class TestEventRunIdStamping:
    """Verify emit() stamps run_id from the current context."""

    def test_emit_stamps_run_id(self) -> None:
        """Events emitted inside run_scope carry the run_id."""
        received: list[BaseEvent] = []

        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(_SimpleEvent)
            def handler(source: Any, event: _SimpleEvent) -> None:
                received.append(event)

            with run_scope("stamp-test") as rid:
                crewai_event_bus.emit(self, _SimpleEvent())

            crewai_event_bus.flush()
            assert len(received) == 1
            assert received[0].run_id == "stamp-test"

    def test_emit_without_run_scope_has_none_run_id(self) -> None:
        """Events emitted outside run_scope have run_id=None."""
        received: list[BaseEvent] = []

        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(_SimpleEvent)
            def handler(source: Any, event: _SimpleEvent) -> None:
                received.append(event)

            crewai_event_bus.emit(self, _SimpleEvent())
            crewai_event_bus.flush()
            assert len(received) == 1
            assert received[0].run_id is None

    def test_explicit_run_id_not_overwritten(self) -> None:
        """If an event already has a run_id set, emit() does not overwrite it."""
        received: list[BaseEvent] = []

        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(_SimpleEvent)
            def handler(source: Any, event: _SimpleEvent) -> None:
                received.append(event)

            evt = _SimpleEvent(run_id="pre-set")
            with run_scope("should-not-override"):
                crewai_event_bus.emit(self, evt)

            crewai_event_bus.flush()
            assert received[0].run_id == "pre-set"


# ---------------------------------------------------------------------------
# 3. Stream handler isolation
# ---------------------------------------------------------------------------

class TestStreamHandlerIsolation:
    """Verify that stream handlers filter by run_id."""

    def test_handler_receives_matching_run_events(self) -> None:
        """A stream handler with run_id only receives matching events."""
        from crewai.utilities.streaming import _create_stream_handler

        q: queue.Queue[Any] = queue.Queue()
        handler = _create_stream_handler(
            current_task_info={
                "index": 0,
                "name": "test",
                "id": "t1",
                "agent_role": "r",
                "agent_id": "a1",
            },
            sync_queue=q,
            run_id="run-A",
        )

        # Event matching run_id
        evt_a = _make_stream_chunk("hello", run_id="run-A")
        handler(None, evt_a)
        assert not q.empty()
        chunk = q.get_nowait()
        assert chunk.content == "hello"

    def test_handler_ignores_mismatched_run_events(self) -> None:
        """A stream handler with run_id ignores events from other runs."""
        from crewai.utilities.streaming import _create_stream_handler

        q: queue.Queue[Any] = queue.Queue()
        handler = _create_stream_handler(
            current_task_info={
                "index": 0,
                "name": "test",
                "id": "t1",
                "agent_role": "r",
                "agent_id": "a1",
            },
            sync_queue=q,
            run_id="run-A",
        )

        # Event from a different run
        evt_b = _make_stream_chunk("wrong-run", run_id="run-B")
        handler(None, evt_b)
        assert q.empty(), "Handler should not enqueue events from another run"

    def test_handler_without_run_id_receives_all(self) -> None:
        """A handler created without run_id (backwards compat) receives all events."""
        from crewai.utilities.streaming import _create_stream_handler

        q: queue.Queue[Any] = queue.Queue()
        handler = _create_stream_handler(
            current_task_info={
                "index": 0,
                "name": "test",
                "id": "t1",
                "agent_role": "r",
                "agent_id": "a1",
            },
            sync_queue=q,
            # No run_id — backwards-compatible mode
        )

        evt = _make_stream_chunk("any", run_id="run-X")
        handler(None, evt)
        assert not q.empty()

        evt_none = _make_stream_chunk("none")
        handler(None, evt_none)
        assert q.qsize() == 2  # Both accepted


# ---------------------------------------------------------------------------
# 4. Concurrent run isolation (integration)
# ---------------------------------------------------------------------------

class TestConcurrentRunIsolation:
    """Integration test: two concurrent runs should not cross-contaminate."""

    def test_two_concurrent_runs_isolated(self) -> None:
        """Simulate two concurrent runs emitting stream chunks.

        Each run has its own handler+queue. Verify zero cross-contamination.
        """
        from crewai.utilities.streaming import _create_stream_handler

        q_a: queue.Queue[Any] = queue.Queue()
        q_b: queue.Queue[Any] = queue.Queue()

        task_info = {
            "index": 0,
            "name": "task",
            "id": "t1",
            "agent_role": "role",
            "agent_id": "a1",
        }

        handler_a = _create_stream_handler(task_info, q_a, run_id="run-A")
        handler_b = _create_stream_handler(task_info, q_b, run_id="run-B")

        with crewai_event_bus.scoped_handlers():
            crewai_event_bus.register_handler(LLMStreamChunkEvent, handler_a)
            crewai_event_bus.register_handler(LLMStreamChunkEvent, handler_b)

            chunks_a = ["a1", "a2", "a3"]
            chunks_b = ["b1", "b2", "b3"]

            barrier = threading.Barrier(2)

            def run_a() -> None:
                with run_scope("run-A"):
                    barrier.wait()
                    for c in chunks_a:
                        crewai_event_bus.emit(
                            self,
                            _make_stream_chunk(c, call_id="call-a"),
                        )

            def run_b() -> None:
                with run_scope("run-B"):
                    barrier.wait()
                    for c in chunks_b:
                        crewai_event_bus.emit(
                            self,
                            _make_stream_chunk(c, call_id="call-b"),
                        )

            t_a = threading.Thread(target=run_a)
            t_b = threading.Thread(target=run_b)
            t_a.start()
            t_b.start()
            t_a.join(timeout=5)
            t_b.join(timeout=5)

            crewai_event_bus.flush()

            # Collect results
            results_a = []
            while not q_a.empty():
                results_a.append(q_a.get_nowait().content)

            results_b = []
            while not q_b.empty():
                results_b.append(q_b.get_nowait().content)

            # Each queue should only have its own chunks
            assert sorted(results_a) == sorted(chunks_a), (
                f"Run A received unexpected chunks: {results_a}"
            )
            assert sorted(results_b) == sorted(chunks_b), (
                f"Run B received unexpected chunks: {results_b}"
            )

    def test_single_run_regression(self) -> None:
        """Single run (no run_scope) still works — full backwards compatibility."""
        received: list[str] = []

        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(LLMStreamChunkEvent)
            def handler(source: Any, event: LLMStreamChunkEvent) -> None:
                received.append(event.chunk)

            # Emit without run_scope — handler has no run_id filter
            crewai_event_bus.emit(self, _make_stream_chunk("hello"))
            crewai_event_bus.emit(self, _make_stream_chunk("world"))
            crewai_event_bus.flush()

            assert received == ["hello", "world"]

    def test_many_concurrent_runs(self) -> None:
        """Stress test: 10 concurrent runs, each emitting 20 chunks."""
        from crewai.utilities.streaming import _create_stream_handler

        num_runs = 10
        chunks_per_run = 20
        task_info = {
            "index": 0,
            "name": "task",
            "id": "t1",
            "agent_role": "r",
            "agent_id": "a1",
        }

        queues: dict[str, queue.Queue[Any]] = {}
        handlers = []

        with crewai_event_bus.scoped_handlers():
            for i in range(num_runs):
                rid = f"run-{i}"
                q: queue.Queue[Any] = queue.Queue()
                queues[rid] = q
                h = _create_stream_handler(task_info, q, run_id=rid)
                handlers.append(h)
                crewai_event_bus.register_handler(LLMStreamChunkEvent, h)

            barrier = threading.Barrier(num_runs)

            def worker(run_id: str) -> None:
                with run_scope(run_id):
                    barrier.wait()
                    for j in range(chunks_per_run):
                        crewai_event_bus.emit(
                            self,
                            _make_stream_chunk(f"{run_id}-{j}", call_id=run_id),
                        )

            threads = []
            for i in range(num_runs):
                t = threading.Thread(target=worker, args=(f"run-{i}",))
                threads.append(t)
                t.start()

            for t in threads:
                t.join(timeout=10)

            crewai_event_bus.flush()

            # Verify isolation
            for i in range(num_runs):
                rid = f"run-{i}"
                results = []
                while not queues[rid].empty():
                    results.append(queues[rid].get_nowait().content)
                expected = [f"{rid}-{j}" for j in range(chunks_per_run)]
                assert sorted(results) == sorted(expected), (
                    f"{rid} received wrong chunks: {sorted(results)}"
                )
