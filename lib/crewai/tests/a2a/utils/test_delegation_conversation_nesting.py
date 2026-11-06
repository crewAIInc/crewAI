"""Regression tests for the A2A conversation/delegation event scope nesting.

A delegation is one turn inside a conversation, so the turn-1
``A2AConversationStartedEvent`` scope must wrap the
``A2ADelegationStartedEvent`` scope. It used to be emitted after the
delegation start, pushing the conversation scope on top: the delegation's
terminal ``A2ADelegationCompletedEvent`` then popped the conversation scope
(parity warning, wrong ``started_event_id``), the delegation scope leaked,
and the tool layer's conversation-completed event popped the delegation
scope — every id in the chain was off by one on every turn-1 delegation,
success or failure.
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
    A2AConversationStartedEvent,
    A2ADelegationCompletedEvent,
    A2ADelegationStartedEvent,
)

ENDPOINT = "http://localhost:9999/agent-card.json"


def make_agent_card() -> AgentCard:
    return AgentCard(
        name="Test Agent",
        description="Test agent for nesting tests",
        url=ENDPOINT,
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        default_input_modes=["text"],
        default_output_modes=["text"],
        skills=[],
    )


class DelegationEventRecorder:
    def __init__(self):
        self.delegation_started = []
        self.delegation_completed = []
        self.conversation_started = []

    def register(self):
        crewai_event_bus.register_handler(
            A2ADelegationStartedEvent, self._on_delegation_started
        )
        crewai_event_bus.register_handler(
            A2ADelegationCompletedEvent, self._on_delegation_completed
        )
        crewai_event_bus.register_handler(
            A2AConversationStartedEvent, self._on_conversation_started
        )

    def unregister(self):
        crewai_event_bus.off(A2ADelegationStartedEvent, self._on_delegation_started)
        crewai_event_bus.off(A2ADelegationCompletedEvent, self._on_delegation_completed)
        crewai_event_bus.off(A2AConversationStartedEvent, self._on_conversation_started)

    def _on_delegation_started(self, source, event):
        self.delegation_started.append(event)

    def _on_delegation_completed(self, source, event):
        self.delegation_completed.append(event)

    def _on_conversation_started(self, source, event):
        self.conversation_started.append(event)


def drain_scopes():
    leftover = []
    while (popped := pop_event_scope()) is not None:
        leftover.append(popped)
    return leftover


@pytest.fixture
def patched_delegation(monkeypatch):
    """Patch the pre-flight and handler so the delegation runs deterministically."""

    async def fake_fetch(*args: Any, **kwargs: Any) -> AgentCard:
        return make_agent_card()

    class OkHandler:
        async def execute(self, **kwargs: Any) -> Any:
            return {"status": "completed", "result": "remote answer"}

    monkeypatch.setattr(delegation, "_afetch_agent_card_cached", fake_fetch)
    monkeypatch.setattr(delegation, "get_handler", lambda config: OkHandler())


async def run_delegation(conversation_history=None) -> Any:
    return await delegation.aexecute_a2a_delegation(
        endpoint=ENDPOINT,
        auth=None,
        timeout=1,
        task_description="do something remotely",
        conversation_history=conversation_history,
    )


@pytest.mark.asyncio
async def test_turn1_completed_pairs_with_delegation_started(patched_delegation):
    """The turn-1 completed event must close the delegation scope, not the conversation's."""
    recorder = DelegationEventRecorder()
    recorder.register()
    try:
        push_event_scope("task-marker", "task_started")
        result = await run_delegation()
        crewai_event_bus.flush()
    finally:
        recorder.unregister()

    assert result["status"] == "completed"
    assert recorder.conversation_started, "turn 1 must open the conversation scope"
    assert recorder.delegation_started
    assert len(recorder.delegation_completed) == 1
    completed = recorder.delegation_completed[0]
    assert completed.started_event_id == recorder.delegation_started[0].event_id, (
        "the completed event must pair with the delegation's own start event, "
        "not pop the conversation scope above it"
    )
    # The conversation scope outlives the delegation (the tool layer closes it
    # when the conversation ends); the delegation scope must be gone.
    assert drain_scopes() == [
        (recorder.conversation_started[0].event_id, "a2a_conversation_started"),
        ("task-marker", "task_started"),
    ]


@pytest.mark.asyncio
async def test_turn2_emits_no_conversation_started(patched_delegation):
    """Only turn 1 opens the conversation scope."""
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
        await run_delegation(conversation_history=[prior_turn])
        crewai_event_bus.flush()
    finally:
        recorder.unregister()

    assert recorder.conversation_started == []
    assert recorder.delegation_started
    assert recorder.delegation_completed[0].started_event_id == (
        recorder.delegation_started[0].event_id
    )
    assert drain_scopes() == [("task-marker", "task_started")]
