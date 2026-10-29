"""Regression tests for A2A delegation event scope pairing.

``aexecute_a2a_delegation`` wraps the implementation in a blanket except
that emits ``A2ADelegationCompletedEvent(status="failed")`` — a
scope-ending event. The implementation only emits its
``A2ADelegationStartedEvent`` after the pre-flight (agent-card fetch,
auth/extension validation, transport/content negotiation) succeeds, so a
pre-flight failure used to emit an ending event with no opener: it popped
the caller's scope (e.g. the task executing the delegation) and stamped
that scope's id as its ``started_event_id``.
"""

from typing import Any

import pytest

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    Message,
    Part,
    Role,
    TextPart,
)

from crewai.a2a.utils import delegation
from crewai.events.event_context import pop_event_scope, push_event_scope
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.a2a_events import (
    A2ADelegationCompletedEvent,
    A2ADelegationStartedEvent,
)

ENDPOINT = "http://localhost:9999/agent-card.json"


def make_agent_card() -> AgentCard:
    return AgentCard(
        name="Test Agent",
        description="Test agent for delegation pairing tests",
        url=ENDPOINT,
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        default_input_modes=["text"],
        default_output_modes=["text"],
        skills=[],
    )


class DelegationEventRecorder:
    def __init__(self):
        self.started = []
        self.completed = []

    def register(self):
        crewai_event_bus.register_handler(A2ADelegationStartedEvent, self._on_started)
        crewai_event_bus.register_handler(
            A2ADelegationCompletedEvent, self._on_completed
        )

    def unregister(self):
        crewai_event_bus.off(A2ADelegationStartedEvent, self._on_started)
        crewai_event_bus.off(A2ADelegationCompletedEvent, self._on_completed)

    def _on_started(self, source, event):
        self.started.append(event)

    def _on_completed(self, source, event):
        self.completed.append(event)


def drain_scopes():
    leftover = []
    while (popped := pop_event_scope()) is not None:
        leftover.append(popped)
    return leftover


async def run_delegation(conversation_history=None) -> Any:
    return await delegation.aexecute_a2a_delegation(
        endpoint=ENDPOINT,
        auth=None,
        timeout=1,
        task_description="do something remotely",
        conversation_history=conversation_history,
    )


@pytest.mark.asyncio
async def test_preflight_failure_emits_no_delegation_events(monkeypatch):
    """A pre-started failure must not emit the scope-ending completed event."""
    monkeypatch.setattr(
        delegation,
        "_afetch_agent_card_cached",
        _raise_card_fetch_failure,
    )
    recorder = DelegationEventRecorder()
    recorder.register()
    try:
        push_event_scope("task-marker", "task_started")
        with pytest.raises(RuntimeError, match="card fetch failed"):
            await run_delegation()
        crewai_event_bus.flush()
    finally:
        recorder.unregister()

    assert recorder.started == []
    assert recorder.completed == [], (
        "a failure before the delegation started event must not emit the "
        "scope-ending completed event, which would pop the caller's scope"
    )
    assert drain_scopes() == [("task-marker", "task_started")]


async def _raise_card_fetch_failure(*args: Any, **kwargs: Any) -> AgentCard:
    raise RuntimeError("card fetch failed")


@pytest.mark.asyncio
async def test_post_started_failure_pairs_completed_event(monkeypatch):
    """A failure after the started event must still emit one paired failed event.

    Uses a second-turn delegation (a prior user message in the history) so the
    turn-1 ``a2a_conversation_started`` scope is not interleaved: the
    conversation lifecycle belongs to the A2A tool layer, and its interaction
    with delegation events on turn-1 failures is a separate concern.
    """

    async def fake_fetch(*args: Any, **kwargs: Any) -> AgentCard:
        return make_agent_card()

    class ExplodingHandler:
        async def execute(self, **kwargs: Any) -> Any:
            raise RuntimeError("handler exploded")

    monkeypatch.setattr(delegation, "_afetch_agent_card_cached", fake_fetch)
    monkeypatch.setattr(delegation, "get_handler", lambda config: ExplodingHandler())

    prior_turn = Message(
        role=Role.user,
        message_id="prior-message",
        parts=[Part(root=TextPart(kind="text", text="previous turn"))],
        context_id="test-context",
    )

    recorder = DelegationEventRecorder()
    recorder.register()
    try:
        push_event_scope("task-marker", "task_started")
        with pytest.raises(RuntimeError, match="handler exploded"):
            await run_delegation(conversation_history=[prior_turn])
        crewai_event_bus.flush()
    finally:
        recorder.unregister()

    assert recorder.started, "a run that passes pre-flight must emit the started event"
    assert len(recorder.completed) == 1
    completed = recorder.completed[0]
    assert completed.status == "failed"
    assert completed.started_event_id == recorder.started[0].event_id, (
        "the failed completed event must pair with the delegation's own start "
        "event, not pop the caller's scope"
    )
    assert drain_scopes() == [("task-marker", "task_started")]
