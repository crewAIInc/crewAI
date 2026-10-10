"""Event bus behavior on runtimes without threads (Pyodide/WebAssembly)."""

from __future__ import annotations

import asyncio
import sys

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
