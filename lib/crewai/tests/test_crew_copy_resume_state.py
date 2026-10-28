"""Regression tests: Crew.copy() must not inherit checkpoint resume state.

Saving a checkpoint syncs the live crew's private runtime state into its
public checkpoint fields (``_sync_checkpoint_fields``), and a completed
run leaves ``task.output`` set. ``Crew.copy()`` — used internally by
``kickoff_for_each``, ``test`` and ``train`` for fresh runs — carried
both into the copy, so the copy's kickoff resumed the old run instead of
executing: ``_get_execution_start_index`` saw the checkpoint pointer and
the stale outputs and skipped every task.
"""

from typing import Any

import pytest

from crewai import Agent, Crew, Task
from crewai.tasks.task_output import TaskOutput


def _checkpointed_crew() -> tuple[Crew, Task]:
    """One crew that has run once and saved a checkpoint.

    Mirrors what ``_sync_checkpoint_fields`` plus a finished kickoff
    leave on the live object.
    """
    agent = Agent(role="r", goal="g", backstory="b", llm="gpt-4o-mini")
    task = Task(description="d1", expected_output="e1", agent=agent)
    crew = Crew(agents=[agent], tasks=[task], name="C")

    task.output = TaskOutput(description="d1", raw="first-run result", agent="r")
    task.checkpoint_original_description = "d1"
    task.checkpoint_original_expected_output = "e1"
    crew.checkpoint_kickoff_event_id = "evt-1"
    crew.checkpoint_inputs = {"x": 1}
    crew.checkpoint_train = False
    return crew, task


class TestCrewCopyResumesFresh:
    def test_copy_drops_checkpoint_resume_pointers(self) -> None:
        crew, _ = _checkpointed_crew()

        copied = crew.copy()

        assert copied.checkpoint_kickoff_event_id is None
        assert copied.checkpoint_inputs is None
        assert copied.checkpoint_train is None

    def test_copy_task_drops_previous_run_output(self) -> None:
        crew, task = _checkpointed_crew()

        copied = crew.copy()

        assert copied.tasks[0] is not task
        assert copied.tasks[0].output is None
        assert copied.tasks[0].checkpoint_original_description is None
        assert copied.tasks[0].checkpoint_original_expected_output is None

    def test_copy_would_execute_its_tasks_from_the_start(self) -> None:
        """The copy must not resume: with the stale outputs inherited, the
        execution start index would equal len(tasks) and skip everything."""
        crew, _ = _checkpointed_crew()
        copied = crew.copy()

        assert copied._get_execution_start_index(copied.tasks) is None

    def test_plain_copy_still_carries_configuration(self) -> None:
        agent = Agent(role="r", goal="g", backstory="b", llm="gpt-4o-mini")
        task = Task(description="d1", expected_output="e1", agent=agent)
        crew = Crew(agents=[agent], tasks=[task], name="C", max_rpm=42)

        copied = crew.copy()

        assert copied.name == "C"
        assert copied.max_rpm == 42
        assert copied.tasks[0].description == "d1"
