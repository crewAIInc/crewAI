"""Regression tests: emit() must track sync-handler futures when async
handlers are also registered, so flush() waits for them and reports their
exceptions -- matching the behavior of replay(), which tracks unconditionally.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.flow_events import MethodExecutionStartedEvent


def _make_event() -> MethodExecutionStartedEvent:
    return MethodExecutionStartedEvent(
        method_name="m",
        flow_name="F",
        params={},
        state={},
    )


def test_flush_waits_for_pending_sync_handler() -> None:
    sync_finished = threading.Event()
    async_finished = threading.Event()

    def slow_sync_handler(_: Any, __: MethodExecutionStartedEvent) -> None:
        time.sleep(0.3)
        sync_finished.set()

    async def quick_async_handler(_: Any, __: MethodExecutionStartedEvent) -> None:
        async_finished.set()

    with crewai_event_bus.scoped_handlers():
        crewai_event_bus.on(MethodExecutionStartedEvent)(slow_sync_handler)
        crewai_event_bus.on(MethodExecutionStartedEvent)(quick_async_handler)

        future = crewai_event_bus.emit(object(), _make_event())
        assert future is not None
        assert crewai_event_bus.flush(timeout=5.0) is True

        assert async_finished.is_set()
        assert sync_finished.is_set(), (
            "flush() returned True while the sync handler was still pending: "
            "emit() did not track the sync-handler future"
        )


def test_flush_reports_sync_handler_exception() -> None:
    printed: list[str] = []

    def failing_sync_handler(_: Any, __: MethodExecutionStartedEvent) -> None:
        raise RuntimeError("sync boom")

    async def quick_async_handler(_: Any, __: MethodExecutionStartedEvent) -> None:
        return None

    with crewai_event_bus.scoped_handlers():
        crewai_event_bus.on(MethodExecutionStartedEvent)(failing_sync_handler)
        crewai_event_bus.on(MethodExecutionStartedEvent)(quick_async_handler)

        original_print = crewai_event_bus._console.print
        crewai_event_bus._console.print = lambda msg: printed.append(str(msg))  # type: ignore[method-assign]
        try:
            future = crewai_event_bus.emit(object(), _make_event())
            assert future is not None
            assert crewai_event_bus.flush(timeout=5.0) is True
        finally:
            crewai_event_bus._console.print = original_print  # type: ignore[method-assign]

    assert any("sync boom" in msg for msg in printed), (
        "flush() did not observe the sync handler exception: its future was "
        "untracked and the error was silently dropped"
    )
