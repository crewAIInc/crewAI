"""Event bus behavior on runtimes without threads (Pyodide/WebAssembly)."""

from __future__ import annotations

import asyncio
import sys
import time

import pytest

from crewai.events.base_events import BaseEvent
from crewai.events.event_bus import CrewAIEventsBus


class _PingEvent(BaseEvent):
    type: str = "ping"


@pytest.fixture
def wasm_bus(monkeypatch: pytest.MonkeyPatch) -> CrewAIEventsBus:
    monkeypatch.setattr(sys, "platform", "emscripten")
    bus = object.__new__(CrewAIEventsBus)
    bus._initialize()
    return bus


@pytest.mark.asyncio
async def test_async_handlers_run_on_the_host_event_loop(
    wasm_bus: CrewAIEventsBus,
) -> None:
    handled = asyncio.Event()

    @wasm_bus.on(_PingEvent)
    async def on_ping(source: object, event: _PingEvent) -> None:
        handled.set()

    wasm_bus.emit(object(), _PingEvent())

    await asyncio.wait_for(handled.wait(), timeout=5)
    assert wasm_bus._loop is asyncio.get_running_loop()


@pytest.mark.asyncio
async def test_shutdown_leaves_the_host_event_loop_running(
    wasm_bus: CrewAIEventsBus,
) -> None:
    @wasm_bus.on(_PingEvent)
    def on_ping(source: object, event: _PingEvent) -> None:
        pass

    wasm_bus.emit(object(), _PingEvent())

    wasm_bus.shutdown(wait=False)

    assert asyncio.get_running_loop().is_running()
    assert not asyncio.get_running_loop().is_closed()
    assert not wasm_bus._sync_handlers


@pytest.mark.asyncio
async def test_flush_returns_without_blocking_the_host_event_loop(
    wasm_bus: CrewAIEventsBus,
) -> None:
    @wasm_bus.on(_PingEvent)
    async def on_ping(source: object, event: _PingEvent) -> None:
        pass

    future = wasm_bus.emit(object(), _PingEvent())

    started = time.monotonic()
    assert wasm_bus.flush(timeout=5) is False
    assert time.monotonic() - started < 1

    await asyncio.wait_for(asyncio.wrap_future(future), timeout=5)
    assert wasm_bus.flush(timeout=5) is True


@pytest.mark.asyncio
async def test_shutdown_without_wait_cancels_pending_async_handlers(
    wasm_bus: CrewAIEventsBus,
) -> None:
    started, cancelled = asyncio.Event(), asyncio.Event()

    @wasm_bus.on(_PingEvent)
    async def on_ping(source: object, event: _PingEvent) -> None:
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    wasm_bus.emit(object(), _PingEvent())
    await asyncio.wait_for(started.wait(), timeout=5)

    wasm_bus.shutdown(wait=False)

    await asyncio.wait_for(cancelled.wait(), timeout=5)
