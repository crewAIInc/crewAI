"""Tests for self_improve/collector.py.

The collector is exercised through the real event bus inside a
``scoped_handlers()`` block. Events are constructed with
``model_construct`` so we don't need to spin up a real Agent + LLM.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.agent_events import (
    AgentExecutionCompletedEvent,
    AgentExecutionErrorEvent,
    AgentExecutionStartedEvent,
)
from crewai.events.types.tool_usage_events import (
    ToolUsageErrorEvent,
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
)
from crewai.skills.self_improve.collector import TraceCollector
from crewai.skills.self_improve.storage import TraceStore


def _fake_agent(*, agent_id: str = "agent-1", role: str = "researcher"):
    return SimpleNamespace(id=agent_id, role=role, skills=None)


def _fake_task(task_id: str = "task-1", description: str = "do the thing"):
    return SimpleNamespace(id=task_id, description=description)


def _started(agent, task) -> AgentExecutionStartedEvent:
    return AgentExecutionStartedEvent.model_construct(
        agent=agent, task=task, tools=[], task_prompt=task.description
    )


def _completed(agent, task, output: str) -> AgentExecutionCompletedEvent:
    return AgentExecutionCompletedEvent.model_construct(
        agent=agent, task=task, output=output
    )


def _error(agent, task, msg: str) -> AgentExecutionErrorEvent:
    return AgentExecutionErrorEvent.model_construct(agent=agent, task=task, error=msg)


def _tool_started(agent_id: str, name: str, args="q=x") -> ToolUsageStartedEvent:
    return ToolUsageStartedEvent.model_construct(
        agent_id=agent_id, tool_name=name, tool_args=args
    )


def _tool_finished(agent_id: str, name: str, args="q=x") -> ToolUsageFinishedEvent:
    now = datetime.now(UTC)
    return ToolUsageFinishedEvent.model_construct(
        agent_id=agent_id,
        tool_name=name,
        tool_args=args,
        started_at=now,
        finished_at=now,
        output="result",
    )


def _tool_error(agent_id: str, name: str, args="q=x") -> ToolUsageErrorEvent:
    return ToolUsageErrorEvent.model_construct(
        agent_id=agent_id, tool_name=name, tool_args=args, error="boom"
    )


@pytest.fixture
def store(tmp_path: Path) -> TraceStore:
    return TraceStore(root=tmp_path)


def test_collects_full_run_and_persists(store: TraceStore) -> None:
    agent = _fake_agent()
    task = _fake_task()
    collector = TraceCollector(agent, store=store)

    with crewai_event_bus.scoped_handlers():
        collector.attach(crewai_event_bus)
        # Flush between emits so the bus thread pool can't reorder
        # tool events past the completion event in this fast test path.
        crewai_event_bus.emit(None, _started(agent, task))
        crewai_event_bus.flush(timeout=5.0)
        crewai_event_bus.emit(None, _tool_started(agent.id, "search"))
        crewai_event_bus.emit(None, _tool_finished(agent.id, "search"))
        crewai_event_bus.flush(timeout=5.0)
        crewai_event_bus.emit(None, _completed(agent, task, "final answer"))
        crewai_event_bus.flush(timeout=5.0)

    saved = store.list_for_role("researcher")
    assert len(saved) == 1

    trace = store.load(saved[0])
    assert trace.agent_role == "researcher"
    assert trace.task_id == "task-1"
    assert trace.output_summary == "final answer"
    assert trace.outcome == "success"
    assert trace.tool_call_count == 1
    assert trace.tool_error_count == 0


def test_error_path_is_graded_as_failure(store: TraceStore) -> None:
    agent = _fake_agent()
    task = _fake_task()
    collector = TraceCollector(agent, store=store)

    with crewai_event_bus.scoped_handlers():
        collector.attach(crewai_event_bus)
        crewai_event_bus.emit(None, _started(agent, task))
        crewai_event_bus.flush(timeout=5.0)
        crewai_event_bus.emit(None, _tool_error(agent.id, "search"))
        crewai_event_bus.flush(timeout=5.0)
        crewai_event_bus.emit(None, _error(agent, task, "agent crashed"))
        crewai_event_bus.flush(timeout=5.0)

    [path] = store.list_for_role("researcher")
    trace = store.load(path)
    assert trace.outcome == "failure"
    assert trace.error == "agent crashed"
    assert trace.tool_error_count == 1


def test_ignores_events_for_other_agents(store: TraceStore) -> None:
    mine = _fake_agent(agent_id="mine", role="researcher")
    other = _fake_agent(agent_id="other", role="editor")
    task = _fake_task()
    collector = TraceCollector(mine, store=store)

    with crewai_event_bus.scoped_handlers():
        collector.attach(crewai_event_bus)
        crewai_event_bus.emit(None, _started(mine, task))
        crewai_event_bus.flush(timeout=5.0)
        # tool events for some other agent must not pollute our trace
        crewai_event_bus.emit(None, _tool_finished(other.id, "leaked-tool"))
        crewai_event_bus.emit(None, _tool_finished(mine.id, "real-tool"))
        crewai_event_bus.flush(timeout=5.0)
        crewai_event_bus.emit(None, _completed(mine, task, "ok"))
        crewai_event_bus.flush(timeout=5.0)

    [path] = store.list_for_role("researcher")
    trace = store.load(path)
    assert [t.name for t in trace.tool_calls] == ["real-tool"]
