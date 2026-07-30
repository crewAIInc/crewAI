"""Tests for ``CrewAIEventsBus.flush`` waiting on sync handlers.

``flush`` waits on ``_pending_futures``, so a handler whose future was never
tracked is invisible to it. ``emit`` used to track the sync-handler future only
when the event type had no async handlers, so for a mixed event type ``flush``
returned before the sync handlers had run.

Each test asserts on the handler's own completion marker rather than on elapsed
time, and holds the handler open on a gate so the assertion cannot be won by a
race.
"""

import asyncio
import threading

from crewai.events.base_events import BaseEvent
from crewai.events.event_bus import crewai_event_bus


class FlushTestEvent(BaseEvent):
    pass


def _tracked_futures() -> set:
    with crewai_event_bus._futures_lock:
        return set(crewai_event_bus._pending_futures)


def test_flush_waits_for_sync_handlers_alongside_async_handlers() -> None:
    """The regression: a mixed event type still has its sync handlers awaited."""
    gate = threading.Event()
    finished: list[str] = []

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(FlushTestEvent)
        def slow_sync_handler(source: object, event: BaseEvent) -> None:
            gate.wait(timeout=5.0)
            finished.append("sync")

        @crewai_event_bus.on(FlushTestEvent)
        async def async_handler(source: object, event: BaseEvent) -> None:
            finished.append("async")

        crewai_event_bus.emit("test_source", FlushTestEvent(type="mixed"))

        # Release the sync handler only once flush is already blocking, so a
        # pass cannot come from the handler having finished beforehand.
        timer = threading.Timer(0.2, gate.set)
        timer.start()
        try:
            assert crewai_event_bus.flush(timeout=10.0) is True
        finally:
            timer.cancel()
            gate.set()

        assert "sync" in finished


def test_flush_waits_for_sync_handlers_when_there_are_no_async_handlers() -> None:
    """The sync-only path already waited; pin it so the fix cannot regress it."""
    gate = threading.Event()
    finished: list[str] = []

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(FlushTestEvent)
        def slow_sync_handler(source: object, event: BaseEvent) -> None:
            gate.wait(timeout=5.0)
            finished.append("sync")

        crewai_event_bus.emit("test_source", FlushTestEvent(type="sync_only"))

        timer = threading.Timer(0.2, gate.set)
        timer.start()
        try:
            assert crewai_event_bus.flush(timeout=10.0) is True
        finally:
            timer.cancel()
            gate.set()

        assert finished == ["sync"]


def test_emit_tracks_the_sync_future_when_async_handlers_exist() -> None:
    """Both futures are tracked, not just the async one.

    The handlers are held open for the duration so the done-callback that
    discards a completed future cannot drop it before the assertion, and the
    tracked set is compared as a delta because it is shared process-wide.
    """
    gate = threading.Event()

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(FlushTestEvent)
        def slow_sync_handler(source: object, event: BaseEvent) -> None:
            gate.wait(timeout=5.0)

        @crewai_event_bus.on(FlushTestEvent)
        async def slow_async_handler(source: object, event: BaseEvent) -> None:
            await asyncio.get_running_loop().run_in_executor(None, gate.wait, 5.0)

        before = _tracked_futures()
        try:
            crewai_event_bus.emit("test_source", FlushTestEvent(type="mixed"))
            added = _tracked_futures() - before
        finally:
            gate.set()

        assert len(added) == 2
        assert crewai_event_bus.flush(timeout=10.0) is True


def test_emit_returns_the_sync_future_only_when_there_are_no_async_handlers() -> None:
    """The documented return-value contract, unchanged by the tracking fix."""
    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(FlushTestEvent)
        def sync_handler(source: object, event: BaseEvent) -> None:
            return None

        sync_only_future = crewai_event_bus.emit(
            "test_source", FlushTestEvent(type="sync_only")
        )

        assert sync_only_future is not None
        assert sync_only_future.result(timeout=10.0) is None

        @crewai_event_bus.on(FlushTestEvent)
        async def async_handler(source: object, event: BaseEvent) -> None:
            return None

        mixed_future = crewai_event_bus.emit(
            "test_source", FlushTestEvent(type="mixed")
        )

        # The asyncio future for the async half, per the ``emit`` docstring.
        assert mixed_future is not None
        assert mixed_future is not sync_only_future
        assert mixed_future.result(timeout=10.0) is None

