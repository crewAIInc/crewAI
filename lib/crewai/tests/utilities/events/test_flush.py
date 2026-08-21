"""Tests for ``CrewAIEventsBus.flush`` waiting on sync handlers.

``flush`` waits on the futures ``emit`` tracked, so a handler whose future was
never tracked is invisible to it. ``emit`` used to track the sync-handler future
only when the event type had no async handlers, so for a mixed event type
``flush`` returned before the sync handlers had run.

Each test asserts through ``flush``'s own contract -- ``False`` on timeout,
``True`` when every handler finished -- rather than on elapsed time or on the
bus's internal future set. A handler that is *still gated* must make ``flush``
time out; only then does releasing the gate and flushing again prove that what
``flush`` waited for was the handler and not the clock.
"""

import asyncio
import threading

from crewai.events.base_events import BaseEvent
from crewai.events.event_bus import crewai_event_bus


class FlushTestEvent(BaseEvent):
    pass


def test_flush_blocks_on_a_sync_handler_alongside_async_handlers() -> None:
    """The regression: a mixed event type still has its sync handlers awaited.

    The async handler is allowed to finish first, so a ``flush`` that timed out
    merely because the async half was still scheduling would not read as a pass.
    From that point the only unfinished handler is the gated sync one.
    """
    gate = threading.Event()
    async_done = threading.Event()
    finished: list[str] = []

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(FlushTestEvent)
        def gated_sync_handler(source: object, event: BaseEvent) -> None:
            gate.wait(timeout=10.0)
            finished.append("sync")

        @crewai_event_bus.on(FlushTestEvent)
        async def async_handler(source: object, event: BaseEvent) -> None:
            finished.append("async")
            async_done.set()

        crewai_event_bus.emit("test_source", FlushTestEvent(type="mixed"))

        try:
            assert async_done.wait(10.0), "the async handler never ran"
            assert finished == ["async"]

            # The sync handler is the only thing outstanding, and it cannot
            # finish. Before the fix its future was never tracked, so flush
            # found nothing to wait on and returned True here.
            assert crewai_event_bus.flush(timeout=1.0) is False
        finally:
            gate.set()

        assert crewai_event_bus.flush(timeout=10.0) is True
        assert finished == ["async", "sync"]


def test_flush_blocks_on_a_sync_handler_when_there_are_no_async_handlers() -> None:
    """The sync-only path already waited; pin it so the fix cannot regress it."""
    gate = threading.Event()
    finished: list[str] = []

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(FlushTestEvent)
        def gated_sync_handler(source: object, event: BaseEvent) -> None:
            gate.wait(timeout=10.0)
            finished.append("sync")

        crewai_event_bus.emit("test_source", FlushTestEvent(type="sync_only"))

        try:
            assert crewai_event_bus.flush(timeout=1.0) is False
            assert finished == []
        finally:
            gate.set()

        assert crewai_event_bus.flush(timeout=10.0) is True
        assert finished == ["sync"]


def test_emit_returns_the_sync_future_only_when_there_are_no_async_handlers() -> None:
    """The documented return-value contract, unchanged by the tracking fix.

    The mixed-case return value is checked by gating the sync handler: the
    asyncio future for the async half resolves while the sync one cannot, so a
    future that resolves here is necessarily not the sync future. Comparing it
    against the earlier sync-only future would not rule out an implementation
    that returned the *current* mixed sync future.
    """
    gate = threading.Event()

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(FlushTestEvent)
        def gated_sync_handler(source: object, event: BaseEvent) -> None:
            gate.wait(timeout=10.0)

        @crewai_event_bus.on(FlushTestEvent)
        async def async_handler(source: object, event: BaseEvent) -> None:
            return None

        mixed_future = crewai_event_bus.emit(
            "test_source", FlushTestEvent(type="mixed")
        )

        try:
            assert mixed_future is not None
            # Resolves with the sync handler still gated, per the emit docstring.
            assert mixed_future.result(timeout=10.0) is None
        finally:
            gate.set()

        assert crewai_event_bus.flush(timeout=10.0) is True

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(FlushTestEvent)
        def sync_handler(source: object, event: BaseEvent) -> None:
            return None

        sync_only_future = crewai_event_bus.emit(
            "test_source", FlushTestEvent(type="sync_only")
        )

        assert sync_only_future is not None
        assert sync_only_future.result(timeout=10.0) is None


def test_flush_waits_for_a_gated_async_handler_too() -> None:
    """The async half stays tracked, so the fix cannot trade one gap for another."""
    gate = threading.Event()

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(FlushTestEvent)
        async def gated_async_handler(source: object, event: BaseEvent) -> None:
            await asyncio.get_running_loop().run_in_executor(None, gate.wait, 10.0)

        crewai_event_bus.emit("test_source", FlushTestEvent(type="async_only"))

        try:
            assert crewai_event_bus.flush(timeout=1.0) is False
        finally:
            gate.set()

        assert crewai_event_bus.flush(timeout=10.0) is True
