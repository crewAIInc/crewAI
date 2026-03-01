"""Regression tests for deterministic task execution ordering.

These tests ensure that tasks with the same implicit priority (all tasks)
are always dispatched in their insertion order — the exact sequence the
user passed to ``Crew(tasks=[...])``.

See: https://github.com/crewAIInc/crewAI/issues/4664
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from crewai.agent import Agent
from crewai.crew import Crew
from crewai.process import Process
from crewai.task import Task
from crewai.tasks.task_output import TaskOutput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_TASKS = 6
ITERATIONS = 5  # run multiple times to surface any non-determinism


def _make_agent(role: str = "Worker") -> Agent:
    return Agent(
        role=role,
        goal="Complete tasks.",
        backstory="You are a diligent worker.",
        allow_delegation=False,
    )


def _make_tasks(agent: Agent, n: int = N_TASKS) -> list[Task]:
    return [
        Task(
            description=f"Task {i}",
            expected_output=f"Output {i}",
            agent=agent,
        )
        for i in range(n)
    ]


def _mock_task_output(desc: str = "mock") -> TaskOutput:
    return TaskOutput(
        description=desc,
        raw="mocked output",
        agent="mocked agent",
        messages=[],
    )


# ---------------------------------------------------------------------------
# Tests: _execution_index stamping
# ---------------------------------------------------------------------------


class TestExecutionIndexStamping:
    """Verify that Crew stamps each task with a stable ``_execution_index``."""

    def test_tasks_receive_execution_index_on_construction(self):
        agent = _make_agent()
        tasks = _make_tasks(agent)
        crew = Crew(agents=[agent], tasks=tasks, process=Process.sequential)

        for idx, task in enumerate(crew.tasks):
            assert task._execution_index == idx, (
                f"Task '{task.description}' should have _execution_index={idx}, "
                f"got {task._execution_index}"
            )

    def test_execution_index_matches_insertion_order(self):
        """Build the crew multiple times and verify indices are stable."""
        agent = _make_agent()

        for _ in range(ITERATIONS):
            tasks = _make_tasks(agent)
            crew = Crew(agents=[agent], tasks=tasks, process=Process.sequential)

            indices = [t._execution_index for t in crew.tasks]
            assert indices == list(range(N_TASKS))

    def test_single_task_gets_index_zero(self):
        agent = _make_agent()
        task = Task(
            description="Only task",
            expected_output="Only output",
            agent=agent,
        )
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
        assert crew.tasks[0]._execution_index == 0

    def test_execution_index_preserved_after_copy(self):
        agent = _make_agent()
        tasks = _make_tasks(agent)
        crew = Crew(agents=[agent], tasks=tasks, process=Process.sequential)

        copied = crew.copy()
        for idx, task in enumerate(copied.tasks):
            assert task._execution_index == idx


# ---------------------------------------------------------------------------
# Tests: deterministic dispatch order (sync)
# ---------------------------------------------------------------------------


class TestDeterministicSyncOrder:
    """Verify that ``_execute_tasks`` dispatches tasks in insertion order."""

    def test_sequential_dispatch_order_is_stable(self):
        """Run the crew multiple times and record the order tasks are dispatched."""
        agent = _make_agent()
        tasks = _make_tasks(agent)

        mock_output = _mock_task_output()
        for t in tasks:
            t.output = mock_output

        crew = Crew(agents=[agent], tasks=tasks, process=Process.sequential)

        for _ in range(ITERATIONS):
            dispatch_order: list[str] = []
            original_execute_sync = Task.execute_sync

            def tracking_execute_sync(self_task, *args, **kwargs):
                dispatch_order.append(self_task.description)
                return mock_output

            with patch.object(Task, "execute_sync", tracking_execute_sync):
                crew.kickoff()

            expected = [f"Task {i}" for i in range(N_TASKS)]
            assert dispatch_order == expected, (
                f"Expected dispatch order {expected}, got {dispatch_order}"
            )

    def test_many_same_description_tasks_preserve_order(self):
        """Tasks with identical descriptions must still keep insertion order."""
        agent = _make_agent()
        tasks = [
            Task(
                description="Identical task",
                expected_output=f"Output {i}",
                agent=agent,
            )
            for i in range(N_TASKS)
        ]

        mock_output = _mock_task_output()
        for t in tasks:
            t.output = mock_output

        crew = Crew(agents=[agent], tasks=tasks, process=Process.sequential)

        dispatch_indices: list[int | None] = []

        def tracking_execute_sync(self_task, *args, **kwargs):
            dispatch_indices.append(self_task._execution_index)
            return mock_output

        with patch.object(Task, "execute_sync", tracking_execute_sync):
            crew.kickoff()

        assert dispatch_indices == list(range(N_TASKS))


# ---------------------------------------------------------------------------
# Tests: deterministic dispatch order (async)
# ---------------------------------------------------------------------------


class TestDeterministicAsyncOrder:
    """Verify that ``_aexecute_tasks`` dispatches tasks in insertion order."""

    @pytest.mark.asyncio
    async def test_async_dispatch_order_is_stable(self):
        """Run the async crew multiple times and verify dispatch order."""
        agent = _make_agent()
        tasks = _make_tasks(agent)

        mock_output = _mock_task_output()
        for t in tasks:
            t.output = mock_output

        crew = Crew(agents=[agent], tasks=tasks, process=Process.sequential)

        for _ in range(ITERATIONS):
            dispatch_order: list[str] = []

            async def tracking_aexecute_sync(self_task, *args, **kwargs):
                dispatch_order.append(self_task.description)
                return mock_output

            with patch.object(Task, "aexecute_sync", tracking_aexecute_sync):
                await crew.akickoff()

            expected = [f"Task {i}" for i in range(N_TASKS)]
            assert dispatch_order == expected, (
                f"Expected dispatch order {expected}, got {dispatch_order}"
            )


# ---------------------------------------------------------------------------
# Tests: task_outputs ordering matches task insertion order
# ---------------------------------------------------------------------------


class TestOutputOrdering:
    """Verify that ``CrewOutput.tasks_output`` preserves insertion order."""

    def test_tasks_output_order_matches_insertion_order(self):
        agent = _make_agent()
        tasks = _make_tasks(agent)

        outputs = [
            TaskOutput(
                description=f"Task {i}",
                raw=f"result {i}",
                agent="Worker",
                messages=[],
            )
            for i in range(N_TASKS)
        ]

        call_index = {"idx": 0}

        def ordered_execute_sync(self_task, *args, **kwargs):
            idx = call_index["idx"]
            call_index["idx"] += 1
            self_task.output = outputs[idx]
            return outputs[idx]

        for t in tasks:
            t.output = outputs[0]  # ensure output is set for validation

        crew = Crew(agents=[agent], tasks=tasks, process=Process.sequential)

        with patch.object(Task, "execute_sync", ordered_execute_sync):
            result = crew.kickoff()

        for i, task_output in enumerate(result.tasks_output):
            assert task_output.raw == f"result {i}", (
                f"tasks_output[{i}] should have raw='result {i}', "
                f"got '{task_output.raw}'"
            )
