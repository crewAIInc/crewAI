"""Regression tests: a failed sync kickoff must stop its sibling threads.

``Task.execute_async`` runs the task in a daemon thread and
``Crew._process_async_tasks`` retrieves the futures in order, so the first
failure used to propagate out of ``kickoff()`` while the siblings carried on:
finishing their agent calls, firing user callbacks and emitting
``TaskCompletedEvent`` for a run the caller had already been told failed.

A thread cannot be preempted, and ``Future.cancel()`` is not a way out either —
the bare future is never moved to RUNNING by an executor, so cancelling it
would succeed and then make the worker thread's ``set_result()`` raise
``InvalidStateError``. The crew therefore signals the task, which stops at the
next boundary it controls.
"""

import threading
import time
from typing import Any
from unittest.mock import patch

import pytest

from crewai.agent import Agent
from crewai.crew import Crew
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.task_events import TaskCompletedEvent
from crewai.task import Task, TaskCancelledError


@pytest.fixture
def researcher() -> Agent:
    return Agent(role="researcher", goal="g", backstory="b")


CALLBACK_CALLS: list[str] = []


def _record_callback(output: Any) -> None:
    """Module-level so Task does not warn about an unserializable callback."""
    CALLBACK_CALLS.append(output.raw)


def test_failed_sync_kickoff_stops_sibling_success_side_effects(
    researcher: Agent,
) -> None:
    """The sibling must not report success for a run that already failed."""
    CALLBACK_CALLS.clear()
    completed: list[str] = []

    @crewai_event_bus.on(TaskCompletedEvent)
    def on_completed(source: Any, event: TaskCompletedEvent) -> None:
        completed.append(event.task.description)

    task_a = Task(
        description="A", expected_output="x", agent=researcher, async_execution=True
    )
    task_b = Task(
        description="B",
        expected_output="x",
        agent=researcher,
        async_execution=True,
        callback=_record_callback,
    )
    # A crew may not end on two async tasks; the trailing sync task is what
    # drains the futures, and it never runs because the drain raises.
    task_c = Task(description="C", expected_output="x", agent=researcher)
    crew = Crew(agents=[researcher], tasks=[task_a, task_b, task_c])

    def fake_execute_task(self: Agent, task: Task, context=None, tools=None) -> str:
        if task.description == "A":
            raise RuntimeError("task A exploded")
        time.sleep(0.3)  # still in flight when the failure reaches the caller
        return "b result"

    try:
        with patch.object(Agent, "execute_task", fake_execute_task):
            with pytest.raises(RuntimeError, match="task A exploded"):
                crew.kickoff()

        # Wait for the sibling to settle before judging what it did.
        assert task_b._thread is not None, "execute_async should keep the handle"
        task_b._thread.join(timeout=30)
        assert not task_b._thread.is_alive()
        crewai_event_bus.flush()

        assert CALLBACK_CALLS == [], "sibling fired its callback after the failure"
        assert "B" not in completed, (
            "sibling reported completion for a run that already failed"
        )
    finally:
        crewai_event_bus.off(TaskCompletedEvent, on_completed)


def test_cancellation_during_a_guardrail_prevents_the_retry(
    researcher: Agent,
) -> None:
    """A blocked guardrail must not go on to spend another agent run."""
    agent_calls: list[str] = []
    in_guardrail = threading.Event()
    release_guardrail = threading.Event()

    def blocking_guardrail(output: Any) -> tuple[bool, str]:
        in_guardrail.set()
        release_guardrail.wait(30)
        return False, "blocked"

    def fake_execute_task(self: Agent, task: Task, context=None, tools=None) -> str:
        agent_calls.append(task.description)
        return "b result"

    task = Task(
        description="B",
        expected_output="x",
        agent=researcher,
        guardrail=blocking_guardrail,
        guardrail_max_retries=2,
    )

    with patch.object(Agent, "execute_task", fake_execute_task):
        future = task.execute_async()
        assert in_guardrail.wait(30), "the guardrail should have been reached"
        task._request_cancellation()
        release_guardrail.set()

        with pytest.raises(TaskCancelledError):
            future.result(timeout=30)

    assert agent_calls == ["B"], "the guardrail retry started another agent run"


def test_a_reused_async_task_starts_uncancelled(researcher: Agent) -> None:
    """The signal belongs to one run, so a reused task is not born cancelled."""
    task = Task(description="B", expected_output="x", agent=researcher)

    def fake_execute_task(self: Agent, task: Task, context=None, tools=None) -> str:
        return "ok"

    with patch.object(Agent, "execute_task", fake_execute_task):
        task._request_cancellation()
        output = task.execute_async().result(timeout=30)

    assert output.raw == "ok"
