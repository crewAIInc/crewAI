"""Tests for sync-handler future tracking in ``CrewAIEventsBus.emit`` / ``flush``."""

from __future__ import annotations

import threading
import time
from typing import Any

from crewai.events.base_events import BaseEvent
from crewai.events.event_bus import crewai_event_bus


class _MixedHandlersEvent(BaseEvent):
    type: str = "test_mixed_handler_flush"


def test_flush_waits_for_sync_handler_when_async_handler_is_also_registered() -> None:
    """``flush()`` must cover the sync-handler future of a mixed emission."""
    sync_done = threading.Event()

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(_MixedHandlersEvent)
        def _slow_sync_handler(_: Any, __: _MixedHandlersEvent) -> None:
            time.sleep(0.2)
            sync_done.set()

        @crewai_event_bus.on(_MixedHandlersEvent)
        async def _fast_async_handler(_: Any, __: _MixedHandlersEvent) -> None:
            return None

        crewai_event_bus.emit(object(), _MixedHandlersEvent())

        assert crewai_event_bus.flush(timeout=5.0) is True
        assert sync_done.is_set(), "flush() returned before the sync handler finished"


def test_flush_waits_for_sync_handler_without_async_handlers() -> None:
    """Sync-only emissions keep their existing flush() semantics."""
    sync_done = threading.Event()

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(_MixedHandlersEvent)
        def _slow_sync_handler(_: Any, __: _MixedHandlersEvent) -> None:
            time.sleep(0.2)
            sync_done.set()

        crewai_event_bus.emit(object(), _MixedHandlersEvent())

        assert crewai_event_bus.flush(timeout=5.0) is True
        assert sync_done.is_set()
