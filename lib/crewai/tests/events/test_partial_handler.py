"""A sync handler may be a functools.partial, which has no __name__."""

from __future__ import annotations

import functools
from collections.abc import Iterator
from typing import Any

import pytest

from crewai.events.base_events import BaseEvent
from crewai.events.event_bus import CrewAIEventsBus, crewai_event_bus


class PartialHandlerTestEvent(BaseEvent):
    value: int = 0
    type: str = "partial_handler_test_event"


CALLS: list[str] = []


def _boom(source: Any, event: Any) -> None:
    raise ValueError("handler exploded")


def _record(source: Any, event: Any) -> None:
    CALLS.append("called")


@pytest.fixture
def bus() -> Iterator[CrewAIEventsBus]:
    CALLS.clear()
    with crewai_event_bus.scoped_handlers():
        yield crewai_event_bus


def test_emit_survives_a_failing_partial_handler(bus: CrewAIEventsBus) -> None:
    bus.register_handler(PartialHandlerTestEvent, functools.partial(_boom))

    future = bus.emit("test_source", PartialHandlerTestEvent())

    if future is not None:
        future.result(timeout=5.0)


def test_emit_runs_a_partial_handler(bus: CrewAIEventsBus) -> None:
    bus.register_handler(PartialHandlerTestEvent, functools.partial(_record))

    future = bus.emit("test_source", PartialHandlerTestEvent())

    if future is not None:
        future.result(timeout=5.0)

    assert CALLS == ["called"]


def test_replay_survives_a_failing_partial_handler(bus: CrewAIEventsBus) -> None:
    bus.register_handler(PartialHandlerTestEvent, functools.partial(_boom))

    future = bus.replay("test_source", PartialHandlerTestEvent())

    if future is not None:
        future.result(timeout=5.0)
