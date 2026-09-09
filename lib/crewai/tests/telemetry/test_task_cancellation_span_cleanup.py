"""Cancelled async tasks must leave no execution_spans entries.

TaskStartedEvent registers the Task in EventListener.execution_spans. Completed and
ordinary-failure paths emit terminal events that pop that entry. asyncio.CancelledError
inherits from BaseException, not Exception, so it used to bypass TaskFailedEvent and
leak a strong reference to the Task (and its Agent/Crew graph).

See https://github.com/crewAIInc/crewAI/issues/7351
"""

from __future__ import annotations

import asyncio
import gc
import threading
import weakref
from unittest.mock import patch

import pytest

from crewai import Agent, Crew, Task
from crewai.events.event_bus import crewai_event_bus
from crewai.events.event_listener import EventListener
from crewai.telemetry import Telemetry


@pytest.fixture
def clean_event_listener():
    """Reset bus and listener singletons so span map state does not leak across tests."""

    def _reset() -> None:
        with crewai_event_bus._rwlock.w_locked():
            crewai_event_bus._sync_handlers.clear()
            crewai_event_bus._async_handlers.clear()
        Telemetry._instance = None
        EventListener._instance = None
        if hasattr(Telemetry, "_lock"):
            Telemetry._lock = threading.Lock()

    _reset()
    listener = EventListener()
    yield listener
    _reset()


async def _cancelled_aexecute(self, *args, **kwargs):
    raise asyncio.CancelledError()


@pytest.mark.asyncio
async def test_cancelled_async_task_clears_execution_spans(clean_event_listener):
    """Cancellation must emit a terminal path and release retained task graphs."""
    listener = clean_event_listener
    listener.execution_spans.clear()

    task_refs: list[weakref.ref] = []
    agent_refs: list[weakref.ref] = []
    crew_refs: list[weakref.ref] = []

    with patch.object(Agent, "aexecute_task", _cancelled_aexecute):
        for _ in range(5):
            agent = Agent(
                role="test",
                goal="test",
                backstory="test",
                llm="gpt-4o-mini",
            )
            task = Task(
                description="test",
                expected_output="test",
                agent=agent,
            )
            crew = Crew(agents=[agent], tasks=[task], tracing=False)
            task_refs.append(weakref.ref(task))
            agent_refs.append(weakref.ref(agent))
            crew_refs.append(weakref.ref(crew))

            with pytest.raises(asyncio.CancelledError):
                await crew.akickoff()

            del crew, task, agent

    gc.collect()

    assert len(listener.execution_spans) == 0
    assert sum(ref() is not None for ref in task_refs) == 0
    assert sum(ref() is not None for ref in agent_refs) == 0
    assert sum(ref() is not None for ref in crew_refs) == 0


@pytest.mark.asyncio
async def test_async_cancelled_error_emits_task_failed_event(clean_event_listener):
    """Producer must emit TaskFailedEvent for CancelledError so the listener can pop."""
    from crewai.events.types.task_events import TaskFailedEvent

    agent = Agent(
        role="tester",
        goal="cancel",
        backstory="exists only to raise CancelledError",
        llm="gpt-4o-mini",
    )
    task = Task(
        description="a task that is cancelled",
        expected_output="nothing",
        agent=agent,
    )

    captured: list[TaskFailedEvent] = []

    def _record(_source, event):
        if isinstance(event, TaskFailedEvent):
            captured.append(event)
        return None

    with patch.object(crewai_event_bus, "emit", side_effect=_record):
        with patch.object(Agent, "aexecute_task", _cancelled_aexecute):
            with pytest.raises(asyncio.CancelledError):
                await task._aexecute_core(None, None, None)

    assert len(captured) == 1
    assert captured[0].error_type is asyncio.CancelledError