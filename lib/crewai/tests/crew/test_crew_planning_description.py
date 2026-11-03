"""Crew planning must not accumulate plan text across kickoffs."""

from unittest.mock import patch

from crewai.agent import Agent
from crewai.crew import Crew
from crewai.crews.utils import prepare_kickoff
from crewai.task import Task
from crewai.utilities.planning_handler import (
    CrewPlanner,
    PlannerTaskPydanticOutput,
    PlanPerTask,
)

PLAN = "PLAN: research the topic, then write it up."


def _crew() -> tuple[Crew, Task]:
    agent = Agent(
        role="Researcher",
        goal="Research",
        backstory="A researcher.",
        llm="gpt-4o-mini",
    )
    task = Task(
        description="Write a report.",
        expected_output="A report.",
        agent=agent,
    )
    return Crew(agents=[agent], tasks=[task], planning=True), task


def _fake_plan(self, summaries: list[str] | None = None):
    if summaries is not None:
        summaries.append(self._create_tasks_summary())
    return PlannerTaskPydanticOutput(
        list_of_plans_per_task=[
            PlanPerTask(task_number=1, task="Write the report", plan=PLAN),
        ]
    )


def test_planning_does_not_duplicate_description_across_kickoffs() -> None:
    crew, task = _crew()

    with patch.object(CrewPlanner, "_handle_crew_planning", _fake_plan):
        prepare_kickoff(crew, None)
        assert task.description == f"Write a report.{PLAN}"

        prepare_kickoff(crew, None)

    assert task.description == f"Write a report.{PLAN}"


def test_planner_does_not_see_the_previous_kickoffs_plan() -> None:
    crew, _ = _crew()
    summaries: list[str] = []

    with patch.object(
        CrewPlanner,
        "_handle_crew_planning",
        lambda self: _fake_plan(self, summaries),
    ):
        prepare_kickoff(crew, None)
        prepare_kickoff(crew, None)

    assert len(summaries) == 2
    assert PLAN not in summaries[0]
    assert PLAN not in summaries[1]
