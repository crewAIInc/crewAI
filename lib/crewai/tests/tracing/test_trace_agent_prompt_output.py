"""The agent_execution_* trace payloads must carry the prompt and the output.

Both types are in ``TraceCollectionListener.complex_events``, so their payload
is hand-built by ``_build_event_data`` rather than serialized from the event:
a bus field reaches the trace only if it is named there. ``task_prompt`` and
``output`` are required on the bus events but were dropped, leaving a trace
that said which agent ran without saying what it was asked or what it said.
"""

from __future__ import annotations

from crewai import Agent
from crewai.events.listeners.tracing.trace_listener import TraceCollectionListener
from crewai.events.types.agent_events import (
    AgentExecutionCompletedEvent,
    AgentExecutionStartedEvent,
)


# Long and non-repeating, so a truncated or elided copy cannot compare equal.
LONG_TEXT = "".join(f"paragraph {i}: the quick brown fox jumps over the lazy dog\n" for i in range(400))
assert len(LONG_TEXT) > 20_000


def _listener() -> TraceCollectionListener:
    """A bare listener: `_build_event_data` needs no batch manager or bus."""
    return TraceCollectionListener.__new__(TraceCollectionListener)


def _agent() -> Agent:
    return Agent(
        role="Researcher",
        goal="Find things out",
        backstory="Curious by nature",
        llm="openai/gpt-4o-mini",
    )


def test_started_payload_carries_the_whole_task_prompt() -> None:
    agent = _agent()
    event = AgentExecutionStartedEvent(
        agent=agent, task=None, tools=None, task_prompt=LONG_TEXT
    )

    data = _listener()._build_event_data("agent_execution_started", event, agent)

    assert data["task_prompt"] == LONG_TEXT
    assert data["agent_role"] == "Researcher"


def test_completed_payload_carries_the_whole_output() -> None:
    agent = _agent()
    event = AgentExecutionCompletedEvent(agent=agent, task=None, output=LONG_TEXT)

    data = _listener()._build_event_data("agent_execution_completed", event, agent)

    assert data["output"] == LONG_TEXT
    assert data["agent_role"] == "Researcher"
