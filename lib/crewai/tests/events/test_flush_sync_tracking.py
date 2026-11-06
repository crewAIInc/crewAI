"""flush()/shutdown visibility of sync-handler futures when async handlers coexist.

Regression test for #8949: emit() only tracked the sync-handler future when no
async handlers were registered, so flush() could return True while a sync
handler was still running. replay() already tracked unconditionally.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from crewai.events.base_events import BaseEvent
from crewai.events.event_bus import crewai_event_bus


class _MixedHandlersEvent(BaseEvent):
    type: str = "mixed_handlers_flush_test"


class TestFlushTracksSyncFuturesWithAsyncHandlers:
    def test_flush_waits_for_sync_handlers_when_async_handlers_exist(self) -> None:
        sync_done = threading.Event()

        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(_MixedHandlersEvent)
            def _slow_sync_handler(_: Any, __: _MixedHandlersEvent) -> None:
                time.sleep(1.0)
                sync_done.set()

            @crewai_event_bus.on(_MixedHandlersEvent)
            async def _fast_async_handler(_: Any, __: _MixedHandlersEvent) -> None:
                return None

            future = crewai_event_bus.emit(object(), _MixedHandlersEvent())
            assert future is not None

            assert crewai_event_bus.flush(timeout=10.0) is True
            assert sync_done.is_set(), (
                "flush() returned True before the sync handler finished"
            )
