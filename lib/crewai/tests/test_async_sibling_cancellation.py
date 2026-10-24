"""Regression tests: failed async execution must cancel sibling tasks.

When one async task (or one crew in a for-each run) fails, the remaining
in-flight siblings used to be abandoned — never cancelled, never awaited —
and kept running in the background after the caller had already received
the error.
"""

import asyncio
from typing import Any
from unittest.mock import patch

import pytest

from crewai.agent import Agent
from crewai.crew import Crew
from crewai.crews.utils import run_for_each_async
from crewai.task import Task
from crewai.tasks.task_output import TaskOutput


@pytest.fixture
def researcher() -> Agent:
    return Agent(role="researcher", goal="g", backstory="b")


def _leftover_tasks() -> list[asyncio.Task[Any]]:
    return [
        t
        for t in asyncio.all_tasks()
        if t is not asyncio.current_task() and not t.done()
    ]


@pytest.mark.asyncio
async def test_async_task_failure_cancels_sibling_tasks(researcher: Agent):
    """A failing async task must cancel its still-running async siblings."""
    b_finished = asyncio.Event()
    b_cancelled = asyncio.Event()

    task_a = Task(
        description="A", expected_output="x", agent=researcher, async_execution=True
    )
    task_b = Task(
        description="B", expected_output="x", agent=researcher, async_execution=True
    )
    task_c = Task(description="C", expected_output="x", agent=researcher)

    crew = Crew(agents=[researcher], tasks=[task_a, task_b, task_c])

    async def fake_aexecute_sync(self: Task, *_args: Any, **_kwargs: Any) -> TaskOutput:
        if self.description == "A":
            raise RuntimeError("task A exploded")
        if self.description == "B":
            try:
                await asyncio.sleep(2.0)
                b_finished.set()
            except asyncio.CancelledError:
                b_cancelled.set()
                raise
            return TaskOutput(description="B", raw="b", agent="researcher")
        return TaskOutput(description="C", raw="c", agent="researcher")

    with patch.object(Task, "aexecute_sync", fake_aexecute_sync):
        with pytest.raises(RuntimeError, match="task A exploded"):
            await crew.akickoff()

    assert b_cancelled.is_set(), "sibling task B should have been cancelled"
    assert not b_finished.is_set(), (
        "sibling task B should not have completed in the background"
    )
    assert not _leftover_tasks(), f"leaked background tasks: {_leftover_tasks()}"


@pytest.mark.asyncio
async def test_run_for_each_failure_cancels_remaining_crews():
    """A failing crew in a for-each run must cancel the still-running crews."""
    second_finished = asyncio.Event()
    second_cancelled = asyncio.Event()

    agent = Agent(role="researcher", goal="g", backstory="b")
    task = Task(description="t", expected_output="x", agent=agent)
    crew = Crew(agents=[agent], tasks=[task])

    async def kickoff_fn(
        _crew_copy: Crew, input_data: dict[str, Any]
    ) -> TaskOutput:
        if input_data.get("fail"):
            raise RuntimeError("crew 0 exploded")
        try:
            await asyncio.sleep(2.0)
            second_finished.set()
        except asyncio.CancelledError:
            second_cancelled.set()
            raise
        raise AssertionError("second crew should not have completed")

    with pytest.raises(RuntimeError, match="crew 0 exploded"):
        await run_for_each_async(
            crew, [{"fail": True}, {"fail": False}], kickoff_fn
        )

    assert second_cancelled.is_set(), "second crew should have been cancelled"
    assert not second_finished.is_set(), (
        "second crew should not have completed in the background"
    )
    assert not _leftover_tasks(), f"leaked background tasks: {_leftover_tasks()}"
