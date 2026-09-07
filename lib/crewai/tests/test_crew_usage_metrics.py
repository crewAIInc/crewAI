"""Tests for crew-level token usage aggregation.

``crew.usage_metrics`` accumulates ``LLMCallCompletedEvent`` for the duration
of one kickoff, so usage reflects what that run actually consumed. Reading an
LLM instance's lifetime counters instead both multiplied usage across agents
sharing the instance and carried earlier runs into later ones.

The aggregator is exercised through the real event bus with fabricated events
and explicit crew-context control; no live LLM provider is required.
"""

from __future__ import annotations

import contextvars
from typing import Any
from uuid import uuid4

from opentelemetry import baggage
from opentelemetry.context import attach, detach

from crewai import Agent, Crew, Task
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.llm_events import LLMCallCompletedEvent, LLMCallType
from crewai.utilities.crew.models import CrewContext


def _emit_llm_call(
    *,
    crew_id: str | None,
    total_tokens: int,
    agent_id: str | None = None,
) -> None:
    """Emit one fake ``LLMCallCompletedEvent`` under ``crew_id``'s context.

    Runs in a freshly-copied context so the crew context the bus snapshots at
    emit time is exactly ``crew_id``, mirroring how ``LLM.call`` emits at
    runtime from inside a kickoff.
    """
    usage: dict[str, Any] = {
        "prompt_tokens": total_tokens,
        "completion_tokens": 0,
        "total_tokens": total_tokens,
    }
    event = LLMCallCompletedEvent(
        call_id=str(uuid4()),
        model="gpt-4o-mini",
        response="ok",
        call_type=LLMCallType.LLM_CALL,
        usage=usage,
    )
    if agent_id is not None:
        event.agent_id = agent_id

    ctx = contextvars.copy_context()

    def _emit() -> None:
        token = None
        if crew_id is not None:
            token = attach(
                baggage.set_baggage("crew_context", CrewContext(id=crew_id, key="k"))
            )
        try:
            future = crewai_event_bus.emit(object(), event)
            if future is not None:
                future.result(timeout=5.0)
        finally:
            if token is not None:
                detach(token)

    ctx.run(_emit)


def _crew() -> Crew:
    agent = Agent(role="Role", goal="goal", backstory="backstory", llm="gpt-4o")
    task = Task(description="task", expected_output="out", agent=agent)
    return Crew(agents=[agent], tasks=[task])


def test_usage_sums_every_observed_call_once() -> None:
    """Each completed call contributes exactly once, including agent-less ones."""
    crew = _crew()
    crew_id = str(crew.id)
    crew._attach_usage_listener()
    try:
        _emit_llm_call(crew_id=crew_id, total_tokens=100, agent_id="agent-a")
        _emit_llm_call(crew_id=crew_id, total_tokens=50, agent_id="agent-b")
        # e.g. crew planning, which runs outside any agent
        _emit_llm_call(crew_id=crew_id, total_tokens=25)
    finally:
        crew._detach_usage_listener()

    assert crew.calculate_usage_metrics().total_tokens == 175


def test_agents_sharing_an_llm_are_not_double_counted() -> None:
    """Two agents on one LLM instance report the calls made, not a multiple.

    Lifetime counters on a shared instance previously produced N x usage for
    N agents; per-call accumulation cannot.
    """
    crew = _crew()
    crew_id = str(crew.id)
    crew._attach_usage_listener()
    try:
        _emit_llm_call(crew_id=crew_id, total_tokens=100, agent_id="agent-a")
        _emit_llm_call(crew_id=crew_id, total_tokens=100, agent_id="agent-b")
    finally:
        crew._detach_usage_listener()

    assert crew.calculate_usage_metrics().total_tokens == 200


def test_second_run_excludes_the_previous_run() -> None:
    """A later kickoff reports only its own usage."""
    crew = _crew()
    crew_id = str(crew.id)

    crew._attach_usage_listener()
    try:
        _emit_llm_call(crew_id=crew_id, total_tokens=100, agent_id="agent-a")
    finally:
        crew._detach_usage_listener()
    assert crew.calculate_usage_metrics().total_tokens == 100

    crew._attach_usage_listener()
    try:
        _emit_llm_call(crew_id=crew_id, total_tokens=50, agent_id="agent-a")
    finally:
        crew._detach_usage_listener()

    assert crew.calculate_usage_metrics().total_tokens == 50


def test_calls_from_another_crew_are_ignored() -> None:
    """Usage is scoped to the crew that is running."""
    crew = _crew()
    crew._attach_usage_listener()
    try:
        _emit_llm_call(crew_id=str(crew.id), total_tokens=100, agent_id="agent-a")
        _emit_llm_call(crew_id=str(uuid4()), total_tokens=999, agent_id="agent-z")
        _emit_llm_call(crew_id=None, total_tokens=999, agent_id="agent-z")
    finally:
        crew._detach_usage_listener()

    assert crew.calculate_usage_metrics().total_tokens == 100
