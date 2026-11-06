"""Regression tests for flush() with mixed sync + async handlers.

emit() must track the sync-handler future even when async handlers are also
registered, so flush() ("block until all pending event handlers complete")
waits for the sync handlers too. The sync-only path always tracked its
future; the mixed path dropped it, letting flush() return early.
"""

import threading
import time

from crewai.events.base_events import BaseEvent
from crewai.events.event_bus import crewai_event_bus


class FlushProbeEvent(BaseEvent):
    """Event type local to these tests."""

    type: str = "flush_probe_event"
    source_type: str = "test"


def test_flush_waits_for_sync_handlers_with_async_handlers_registered():
    sync_done = threading.Event()
    async_done = threading.Event()

    def slow_sync_handler(source, event):
        time.sleep(0.5)
        sync_done.set()

    async def fast_async_handler(source, event):
        async_done.set()

    crewai_event_bus.register_handler(FlushProbeEvent, slow_sync_handler)
    crewai_event_bus.register_handler(FlushProbeEvent, fast_async_handler)
    try:
        crewai_event_bus.emit("test-source", FlushProbeEvent())

        completed = crewai_event_bus.flush(timeout=10)

        assert completed, "flush should report all handlers completed"
        assert async_done.is_set()
        assert sync_done.is_set(), (
            "flush() returned before the sync handler finished"
        )
    finally:
        crewai_event_bus.off(FlushProbeEvent, slow_sync_handler)
        crewai_event_bus.off(FlushProbeEvent, fast_async_handler)


def test_flush_timeout_reports_pending_sync_handler():
    sync_done = threading.Event()

    def slow_sync_handler(source, event):
        time.sleep(1.0)
        sync_done.set()

    async def fast_async_handler(source, event):
        return None

    crewai_event_bus.register_handler(FlushProbeEvent, slow_sync_handler)
    crewai_event_bus.register_handler(FlushProbeEvent, fast_async_handler)
    try:
        crewai_event_bus.emit("test-source", FlushProbeEvent())

        completed = crewai_event_bus.flush(timeout=0.2)

        assert not completed, (
            "flush() should not report completion while a sync handler is pending"
        )
        assert not sync_done.is_set()
    finally:
        crewai_event_bus.flush(timeout=10)
        crewai_event_bus.off(FlushProbeEvent, slow_sync_handler)
        crewai_event_bus.off(FlushProbeEvent, fast_async_handler)


def test_flush_waits_for_sync_only_handlers():
    sync_done = threading.Event()

    def slow_sync_handler(source, event):
        time.sleep(0.3)
        sync_done.set()

    crewai_event_bus.register_handler(FlushProbeEvent, slow_sync_handler)
    try:
        crewai_event_bus.emit("test-source", FlushProbeEvent())

        completed = crewai_event_bus.flush(timeout=10)

        assert completed
        assert sync_done.is_set()
    finally:
        crewai_event_bus.off(FlushProbeEvent, slow_sync_handler)
