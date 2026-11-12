"""An early delegation failure must not emit an unpaired terminal event.

``a2a_delegation_completed`` is scope-ending: the event bus pops the scope
stack when it is emitted. Card fetch, auth validation and extension
validation raise inside ``_aexecute_a2a_delegation_impl`` *before*
``A2ADelegationStartedEvent`` is emitted, but the failure handler emitted the
Completed event unconditionally — popping whatever enclosing scope was open
(e.g. the caller's flow/agent frame) instead, corrupting later event
parentage in that task.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import crewai.a2a.utils.delegation as delegation_mod
from crewai.events.event_context import (
    get_current_parent_id,
    pop_event_scope,
    push_event_scope,
)
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.a2a_events import (
    A2ADelegationCompletedEvent,
    A2ADelegationStartedEvent,
)


@pytest.fixture
def captured_events() -> Any:
    events: list[Any] = []

    @crewai_event_bus.on(A2ADelegationStartedEvent)
    def _on_started(_source: Any, event: A2ADelegationStartedEvent) -> None:
        events.append(event)

    @crewai_event_bus.on(A2ADelegationCompletedEvent)
    def _on_completed(_source: Any, event: A2ADelegationCompletedEvent) -> None:
        events.append(event)

    yield events
    crewai_event_bus.off(A2ADelegationStartedEvent, _on_started)
    crewai_event_bus.off(A2ADelegationCompletedEvent, _on_completed)


@pytest.fixture
def fail_card_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    async def boom(**kwargs: Any) -> None:
        raise ConnectionError("no route to host")

    monkeypatch.setattr(delegation_mod, "_afetch_agent_card_cached", boom)


async def _delegation_raising_connection_error() -> None:
    with pytest.raises(ConnectionError):
        await delegation_mod.aexecute_a2a_delegation(
            endpoint="http://remote-agent",
            auth=None,
            timeout=1,
            task_description="do something",
        )


def test_early_failure_emits_no_unpaired_completed(
    fail_card_fetch: None, captured_events: list[Any]
) -> None:
    asyncio.run(_delegation_raising_connection_error())
    crewai_event_bus.flush()

    started = [e for e in captured_events if isinstance(e, A2ADelegationStartedEvent)]
    completed = [
        e for e in captured_events if isinstance(e, A2ADelegationCompletedEvent)
    ]
    assert not started
    for event in completed:
        assert event.started_event_id is not None, (
            "a2a_delegation_completed emitted without a matching "
            "a2a_delegation_started (started_event_id is None)"
        )


def test_early_failure_does_not_steal_the_enclosing_scope(
    fail_card_fetch: None,
) -> None:
    async def run() -> None:
        # The delegation runs inside an enclosing step whose flow_started
        # frame is on the scope stack.
        push_event_scope("parent-flow-id", "flow_started")
        try:
            await _delegation_raising_connection_error()
            assert get_current_parent_id() == "parent-flow-id"
        finally:
            pop_event_scope()

    asyncio.run(run())
