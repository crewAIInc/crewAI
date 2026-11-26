"""Regression tests for agent-reasoning event scope pairing.

Each reasoning attempt (the initial plan plus every refinement) emits its own
``AgentReasoningStartedEvent``, which opens an event scope. A failure during
a refinement used to propagate straight to ``handle_agent_reasoning``'s outer
except, which emits a single ``AgentReasoningFailedEvent``: the refine
attempt's scope was popped but attempt 1's scope leaked, so the next ending
event in the context (e.g. ``task_failed``) popped the leaked reasoning scope
instead of its own opener.
"""

from unittest.mock import MagicMock

import pytest

from crewai.agent.planning_config import PlanningConfig
from crewai.events.event_context import pop_event_scope, push_event_scope
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.reasoning_events import (
    AgentReasoningCompletedEvent,
    AgentReasoningFailedEvent,
    AgentReasoningStartedEvent,
)
from crewai.utilities.reasoning_handler import AgentReasoning

READY_MARK = "READY: I am ready to execute the task."


class ScriptedLLM:
    """Text-path LLM: returns scripted responses or raises on a given call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def supports_function_calling(self):
        return False

    def call(self, messages, **kwargs):
        self.calls += 1
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_agent(llm):
    agent = MagicMock()
    agent.role = "Test Agent"
    agent.goal = "Test goal"
    agent.backstory = "Test backstory"
    agent.verbose = False
    agent.planning_config = PlanningConfig()
    agent.llm = llm
    return agent


def make_handler(llm):
    return AgentReasoning(
        agent=make_agent(llm),
        task=None,
        description="Do the thing",
        expected_output="The thing done",
    )


class ReasoningEventRecorder:
    def __init__(self):
        self.started = []
        self.completed = []
        self.failed = []

    def register(self):
        crewai_event_bus.register_handler(AgentReasoningStartedEvent, self._on_started)
        crewai_event_bus.register_handler(
            AgentReasoningCompletedEvent, self._on_completed
        )
        crewai_event_bus.register_handler(AgentReasoningFailedEvent, self._on_failed)

    def unregister(self):
        crewai_event_bus.off(AgentReasoningStartedEvent, self._on_started)
        crewai_event_bus.off(AgentReasoningCompletedEvent, self._on_completed)
        crewai_event_bus.off(AgentReasoningFailedEvent, self._on_failed)

    def _on_started(self, source, event):
        self.started.append(event)

    def _on_completed(self, source, event):
        self.completed.append(event)

    def _on_failed(self, source, event):
        self.failed.append(event)


def drain_scopes():
    leftover = []
    while (popped := pop_event_scope()) is not None:
        leftover.append(popped)
    return leftover


def test_refine_failure_pairs_every_attempt_scope():
    """A failed refinement must close its own scope, not leak attempt 1's."""
    llm = ScriptedLLM(
        [
            "Step 1: draft. Step 2: review. Not confident yet.",
            RuntimeError("refine call exploded"),
        ]
    )
    handler = make_handler(llm)
    recorder = ReasoningEventRecorder()
    recorder.register()
    try:
        push_event_scope("task-marker", "task_started")
        try:
            with pytest.raises(RuntimeError, match="refine call exploded"):
                handler.handle_agent_reasoning()
        finally:
            crewai_event_bus.flush()
    finally:
        recorder.unregister()

    started_ids = [event.event_id for event in recorder.started]
    failed_ids = [event.started_event_id for event in recorder.failed]
    assert sorted(event.attempt for event in recorder.started) == [1, 2]
    assert set(failed_ids) == set(started_ids), (
        "every started attempt scope must be closed by a failed event, not "
        "just the innermost one"
    )
    assert sorted(event.attempt for event in recorder.failed) == [1, 2]
    assert drain_scopes() == [("task-marker", "task_started")], (
        "the enclosing task scope must survive the reasoning failure"
    )


def test_refine_success_pairs_every_attempt_scope():
    """A successful refinement closes each attempt scope exactly once."""
    llm = ScriptedLLM(
        [
            "Step 1: draft. Step 2: review. Not confident yet.",
            f"Plan finalized. {READY_MARK}",
        ]
    )
    handler = make_handler(llm)
    recorder = ReasoningEventRecorder()
    recorder.register()
    try:
        push_event_scope("task-marker", "task_started")
        handler.handle_agent_reasoning()
        crewai_event_bus.flush()
    finally:
        recorder.unregister()

    started_ids = [event.event_id for event in recorder.started]
    completed_ids = [event.started_event_id for event in recorder.completed]
    assert sorted(event.attempt for event in recorder.started) == [1, 2]
    assert sorted(completed_ids) == sorted(started_ids), (
        "each attempt scope must be closed by exactly one completed event"
    )
    assert recorder.failed == []
    assert drain_scopes() == [("task-marker", "task_started")]
