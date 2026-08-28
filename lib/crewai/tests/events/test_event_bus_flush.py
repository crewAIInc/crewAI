"""Tests for event bus flush() coverage of sync handlers with mixed handler types."""

from __future__ import annotations

import threading
import time
from typing import Any

from crewai.events.base_events import BaseEvent
from crewai.events.event_bus import crewai_event_bus


class _FlushProbeEvent(BaseEvent):
    """Minimal event type used to probe flush() behavior."""


class TestFlushWithMixedHandlers:
    """flush() must wait for sync handlers even when async handlers are registered."""

    def test_flush_waits_for_sync_handler_when_async_handler_registered(self) -> None:
        """A slow sync handler must finish before flush() returns (regression #6745)."""
        sync_done = threading.Event()

        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(_FlushProbeEvent)
            def slow_sync(_: Any, event: _FlushProbeEvent) -> None:
                time.sleep(0.4)
                sync_done.set()

            @crewai_event_bus.on(_FlushProbeEvent)
            async def quick_async(_: Any, event: _FlushProbeEvent) -> None:
                pass

            crewai_event_bus.emit("source", _FlushProbeEvent(type="flush-probe"))

            ok = crewai_event_bus.flush(timeout=5.0)

        assert ok is True
        assert sync_done.is_set(), (
            "flush() returned before the sync handler finished for a mixed event type"
        )

    def test_flush_waits_for_sync_only_handlers(self) -> None:
        """The sync-only path still blocks flush() until the handler finishes."""
        sync_done = threading.Event()

        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(_FlushProbeEvent)
            def slow_sync(_: Any, event: _FlushProbeEvent) -> None:
                time.sleep(0.4)
                sync_done.set()

            crewai_event_bus.emit("source", _FlushProbeEvent(type="flush-probe"))

            ok = crewai_event_bus.flush(timeout=5.0)

        assert ok is True
        assert sync_done.is_set()

    def test_emit_returns_future_for_mixed_handlers(self) -> None:
        """emit() still returns the async future for mixed handler types (contract)."""
        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(_FlushProbeEvent)
            def slow_sync(_: Any, event: _FlushProbeEvent) -> None:
                time.sleep(0.4)

            @crewai_event_bus.on(_FlushProbeEvent)
            async def quick_async(_: Any, event: _FlushProbeEvent) -> None:
                pass

            future = crewai_event_bus.emit("source", _FlushProbeEvent(type="flush-probe"))

            assert future is not None
            assert future.result(timeout=2.0) is None
            assert crewai_event_bus.flush(timeout=5.0) is True
