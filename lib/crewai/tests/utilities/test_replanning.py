"""Tests for the adaptive replanning feature (issue #4983).

Covers:
- ReplanningEvaluator: evaluation of task results against plans
- CrewPlanner._handle_crew_replanning: generating revised plans
- Crew integration: replan_on_failure / max_replans fields and the
  _maybe_replan hook in both sync and async execution paths
- Backwards compatibility: existing crews are unaffected by default
"""

from unittest.mock import MagicMock, patch, call

import pytest

from crewai.agent import Agent
from crewai.crew import Crew
from crewai.task import Task
from crewai.tasks.task_output import TaskOutput
from crewai.utilities.planning_handler import (
    CrewPlanner,
    PlannerTaskPydanticOutput,
    PlanPerTask,
)
from crewai.utilities.replanning_evaluator import (
    ReplanDecision,
    ReplanningEvaluator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agents(n: int = 3) -> list[Agent]:
    return [
        Agent(role=f"Agent {i}", goal=f"Goal {i}", backstory=f"Backstory {i}")
        for i in range(1, n + 1)
    ]


def _make_tasks(agents: list[Agent]) -> list[Task]:
    return [
        Task(
            description=f"Task {i} description",
            expected_output=f"Output {i}",
            agent=agents[i - 1],
        )
        for i in range(1, len(agents) + 1)
    ]


def _task_output(raw: str = "result", agent: str = "agent") -> TaskOutput:
    return TaskOutput(description="desc", agent=agent, raw=raw)


# ---------------------------------------------------------------------------
# ReplanDecision model tests
# ---------------------------------------------------------------------------

class TestReplanDecision:
    def test_replan_decision_defaults(self):
        decision = ReplanDecision(
            should_replan=False,
            reason="All good",
        )
        assert decision.should_replan is False
        assert decision.reason == "All good"
        assert decision.affected_task_numbers == []

    def test_replan_decision_with_affected_tasks(self):
        decision = ReplanDecision(
            should_replan=True,
            reason="Data missing",
            affected_task_numbers=[2, 3],
        )
        assert decision.should_replan is True
        assert decision.affected_task_numbers == [2, 3]


# ---------------------------------------------------------------------------
# ReplanningEvaluator tests
# ---------------------------------------------------------------------------

class TestReplanningEvaluator:
    def test_default_llm(self):
        evaluator = ReplanningEvaluator()
        assert evaluator.llm == "gpt-4o-mini"

    def test_custom_llm(self):
        evaluator = ReplanningEvaluator(llm="gpt-4o")
        assert evaluator.llm == "gpt-4o"

    def test_evaluate_returns_replan_decision_when_deviation_detected(self):
        """When the LLM says replanning is needed, evaluate() returns that."""
        evaluator = ReplanningEvaluator()
        agents = _make_agents(3)
        tasks = _make_tasks(agents)
        output = _task_output("completely unexpected result")

        expected_decision = ReplanDecision(
            should_replan=True,
            reason="Result deviates from plan assumptions",
            affected_task_numbers=[2, 3],
        )

        with patch.object(Task, "execute_sync") as mock_exec:
            mock_exec.return_value = TaskOutput(
                description="eval",
                agent="evaluator",
                pydantic=expected_decision,
            )
            decision = evaluator.evaluate(
                completed_task=tasks[0],
                task_output=output,
                original_plan="Step 1: do X\nStep 2: do Y",
                remaining_tasks=tasks[1:],
            )

        assert decision.should_replan is True
        assert decision.reason == "Result deviates from plan assumptions"
        mock_exec.assert_called_once()

    def test_evaluate_returns_no_replan_when_result_matches(self):
        """When the result matches the plan, no replanning is needed."""
        evaluator = ReplanningEvaluator()
        agents = _make_agents(2)
        tasks = _make_tasks(agents)
        output = _task_output("expected result matching plan")

        expected_decision = ReplanDecision(
            should_replan=False,
            reason="Result aligns with plan",
            affected_task_numbers=[],
        )

        with patch.object(Task, "execute_sync") as mock_exec:
            mock_exec.return_value = TaskOutput(
                description="eval",
                agent="evaluator",
                pydantic=expected_decision,
            )
            decision = evaluator.evaluate(
                completed_task=tasks[0],
                task_output=output,
                original_plan="Step 1: gather data",
                remaining_tasks=tasks[1:],
            )

        assert decision.should_replan is False

    def test_evaluate_fallback_on_bad_output(self):
        """When the LLM returns non-structured output, fallback to no-replan."""
        evaluator = ReplanningEvaluator()
        agents = _make_agents(2)
        tasks = _make_tasks(agents)
        output = _task_output("some result")

        with patch.object(Task, "execute_sync") as mock_exec:
            mock_exec.return_value = TaskOutput(
                description="eval",
                agent="evaluator",
                pydantic=None,  # no structured output
            )
            decision = evaluator.evaluate(
                completed_task=tasks[0],
                task_output=output,
                original_plan="Step 1: do stuff",
                remaining_tasks=tasks[1:],
            )

        assert decision.should_replan is False
        assert "Failed to evaluate" in decision.reason

    def test_summarize_remaining_tasks_empty(self):
        result = ReplanningEvaluator._summarize_remaining_tasks([])
        assert result == "No remaining tasks."

    def test_summarize_remaining_tasks_with_tasks(self):
        agents = _make_agents(2)
        tasks = _make_tasks(agents)
        result = ReplanningEvaluator._summarize_remaining_tasks(tasks)
        assert "Task 1" in result
        assert "Task 2" in result
        assert "Agent 1" in result


# ---------------------------------------------------------------------------
# CrewPlanner._handle_crew_replanning tests
# ---------------------------------------------------------------------------

class TestCrewPlannerReplanning:
    @pytest.fixture
    def planner(self):
        agents = _make_agents(3)
        tasks = _make_tasks(agents)
        return CrewPlanner(tasks=tasks, planning_agent_llm=None)

    def test_handle_crew_replanning_returns_revised_plans(self, planner):
        agents = _make_agents(3)
        tasks = _make_tasks(agents)
        outputs = [_task_output("result 1")]

        revised_plans = PlannerTaskPydanticOutput(
            list_of_plans_per_task=[
                PlanPerTask(task_number=1, task="Task 2", plan="Revised plan for task 2"),
                PlanPerTask(task_number=2, task="Task 3", plan="Revised plan for task 3"),
            ]
        )

        with patch.object(Task, "execute_sync") as mock_exec:
            mock_exec.return_value = TaskOutput(
                description="replan",
                agent="planner",
                pydantic=revised_plans,
            )
            result = planner._handle_crew_replanning(
                completed_tasks=[tasks[0]],
                completed_outputs=outputs,
                remaining_tasks=tasks[1:],
                deviation_reason="Task 1 returned unexpected data",
            )

        assert isinstance(result, PlannerTaskPydanticOutput)
        assert len(result.list_of_plans_per_task) == 2
        mock_exec.assert_called_once()

    def test_handle_crew_replanning_raises_on_bad_output(self, planner):
        agents = _make_agents(3)
        tasks = _make_tasks(agents)
        outputs = [_task_output("result 1")]

        with patch.object(Task, "execute_sync") as mock_exec:
            mock_exec.return_value = TaskOutput(
                description="replan",
                agent="planner",
                pydantic=None,
            )
            with pytest.raises(ValueError, match="Failed to get the Replanning output"):
                planner._handle_crew_replanning(
                    completed_tasks=[tasks[0]],
                    completed_outputs=outputs,
                    remaining_tasks=tasks[1:],
                    deviation_reason="deviation",
                )

    def test_completed_tasks_summary(self):
        agents = _make_agents(2)
        tasks = _make_tasks(agents)
        outputs = [_task_output("result A"), _task_output("result B")]

        summary = CrewPlanner._create_completed_tasks_summary(tasks, outputs)
        assert "result A" in summary
        assert "result B" in summary
        assert "Agent 1" in summary

    def test_completed_tasks_summary_empty(self):
        summary = CrewPlanner._create_completed_tasks_summary([], [])
        assert summary == "No completed tasks."

    def test_tasks_summary_for_remaining(self):
        agents = _make_agents(2)
        tasks = _make_tasks(agents)
        summary = CrewPlanner._create_tasks_summary_for(tasks)
        assert "Task Number 1" in summary
        assert "Task Number 2" in summary

    def test_tasks_summary_for_empty(self):
        summary = CrewPlanner._create_tasks_summary_for([])
        assert summary == "No remaining tasks."


# ---------------------------------------------------------------------------
# Crew field tests (backwards compatibility)
# ---------------------------------------------------------------------------

class TestCrewReplanningFields:
    def test_replan_on_failure_defaults_to_false(self):
        agents = _make_agents(1)
        tasks = _make_tasks(agents)
        crew = Crew(agents=agents, tasks=tasks)
        assert crew.replan_on_failure is False

    def test_max_replans_defaults_to_3(self):
        agents = _make_agents(1)
        tasks = _make_tasks(agents)
        crew = Crew(agents=agents, tasks=tasks)
        assert crew.max_replans == 3

    def test_replan_on_failure_can_be_set(self):
        agents = _make_agents(1)
        tasks = _make_tasks(agents)
        crew = Crew(agents=agents, tasks=tasks, replan_on_failure=True)
        assert crew.replan_on_failure is True

    def test_max_replans_can_be_set(self):
        agents = _make_agents(1)
        tasks = _make_tasks(agents)
        crew = Crew(agents=agents, tasks=tasks, max_replans=5)
        assert crew.max_replans == 5

    def test_max_replans_cannot_be_negative(self):
        agents = _make_agents(1)
        tasks = _make_tasks(agents)
        with pytest.raises(ValueError):
            Crew(agents=agents, tasks=tasks, max_replans=-1)


# ---------------------------------------------------------------------------
# Crew._maybe_replan integration tests
# ---------------------------------------------------------------------------

class TestCrewMaybeReplan:
    def _setup_crew_with_planning(self, n_agents: int = 3) -> tuple[Crew, list[Agent], list[Task]]:
        agents = _make_agents(n_agents)
        tasks = _make_tasks(agents)
        crew = Crew(
            agents=agents,
            tasks=tasks,
            planning=True,
            replan_on_failure=True,
            max_replans=3,
        )
        # Simulate planning having been called
        crew._original_task_descriptions = [t.description for t in tasks]
        crew._replan_count = 0
        # Append a fake plan to each task
        for task in tasks:
            task.description += " [PLAN]"
        return crew, agents, tasks

    def test_maybe_replan_skips_when_planning_disabled(self):
        agents = _make_agents(2)
        tasks = _make_tasks(agents)
        crew = Crew(agents=agents, tasks=tasks, planning=False, replan_on_failure=True)
        crew._original_task_descriptions = [t.description for t in tasks]

        # Should not call evaluator at all
        with patch.object(ReplanningEvaluator, "evaluate") as mock_eval:
            crew._maybe_replan(tasks[0], _task_output(), 0, tasks, [_task_output()])
            mock_eval.assert_not_called()

    def test_maybe_replan_skips_when_replan_on_failure_disabled(self):
        agents = _make_agents(2)
        tasks = _make_tasks(agents)
        crew = Crew(agents=agents, tasks=tasks, planning=True, replan_on_failure=False)
        crew._original_task_descriptions = [t.description for t in tasks]

        with patch.object(ReplanningEvaluator, "evaluate") as mock_eval:
            crew._maybe_replan(tasks[0], _task_output(), 0, tasks, [_task_output()])
            mock_eval.assert_not_called()

    def test_maybe_replan_skips_on_last_task(self):
        crew, agents, tasks = self._setup_crew_with_planning(2)

        with patch.object(ReplanningEvaluator, "evaluate") as mock_eval:
            crew._maybe_replan(tasks[1], _task_output(), 1, tasks, [_task_output()])
            mock_eval.assert_not_called()

    def test_maybe_replan_skips_when_max_replans_reached(self):
        crew, agents, tasks = self._setup_crew_with_planning(3)
        crew._replan_count = 3  # already at max

        with patch.object(ReplanningEvaluator, "evaluate") as mock_eval:
            crew._maybe_replan(tasks[0], _task_output(), 0, tasks, [_task_output()])
            mock_eval.assert_not_called()

    def test_maybe_replan_skips_when_no_plan_text(self):
        agents = _make_agents(3)
        tasks = _make_tasks(agents)
        crew = Crew(
            agents=agents, tasks=tasks,
            planning=True, replan_on_failure=True,
        )
        crew._original_task_descriptions = [t.description for t in tasks]
        crew._replan_count = 0
        # No plan appended — descriptions are unchanged

        with patch.object(ReplanningEvaluator, "evaluate") as mock_eval:
            crew._maybe_replan(tasks[0], _task_output(), 0, tasks, [_task_output()])
            mock_eval.assert_not_called()

    def test_maybe_replan_no_replan_when_evaluator_says_no(self):
        crew, agents, tasks = self._setup_crew_with_planning(3)
        original_desc_1 = tasks[1].description
        original_desc_2 = tasks[2].description

        no_replan = ReplanDecision(
            should_replan=False,
            reason="Result is fine",
            affected_task_numbers=[],
        )

        with patch.object(ReplanningEvaluator, "evaluate", return_value=no_replan):
            with patch.object(CrewPlanner, "_handle_crew_replanning") as mock_handler:
                crew._maybe_replan(tasks[0], _task_output(), 0, tasks, [_task_output()])
                mock_handler.assert_not_called()

        assert crew._replan_count == 0
        # Task descriptions should be unchanged
        assert tasks[1].description == original_desc_1
        assert tasks[2].description == original_desc_2

    def test_maybe_replan_triggers_replanning_and_updates_tasks(self):
        crew, agents, tasks = self._setup_crew_with_planning(3)

        deviation_decision = ReplanDecision(
            should_replan=True,
            reason="Task 1 returned error data",
            affected_task_numbers=[2, 3],
        )

        revised_plans = PlannerTaskPydanticOutput(
            list_of_plans_per_task=[
                PlanPerTask(task_number=1, task="Task 2", plan=" [REVISED PLAN 2]"),
                PlanPerTask(task_number=2, task="Task 3", plan=" [REVISED PLAN 3]"),
            ]
        )

        with patch.object(ReplanningEvaluator, "evaluate", return_value=deviation_decision):
            with patch.object(
                CrewPlanner, "_handle_crew_replanning", return_value=revised_plans
            ):
                crew._maybe_replan(
                    tasks[0], _task_output("bad result"), 0, tasks, [_task_output("bad result")]
                )

        assert crew._replan_count == 1
        # Remaining tasks should have the revised plans
        assert "[REVISED PLAN 2]" in tasks[1].description
        assert "[REVISED PLAN 3]" in tasks[2].description
        # Old plan should be gone (original desc restored + new plan)
        assert tasks[1].description.count("[PLAN]") == 0
        assert tasks[2].description.count("[PLAN]") == 0

    def test_maybe_replan_increments_replan_count_each_time(self):
        crew, agents, tasks = self._setup_crew_with_planning(3)

        deviation = ReplanDecision(
            should_replan=True,
            reason="deviation",
            affected_task_numbers=[2],
        )

        revised = PlannerTaskPydanticOutput(
            list_of_plans_per_task=[
                PlanPerTask(task_number=1, task="T2", plan=" [NEW PLAN]"),
            ]
        )

        with patch.object(ReplanningEvaluator, "evaluate", return_value=deviation):
            with patch.object(CrewPlanner, "_handle_crew_replanning", return_value=revised):
                crew._maybe_replan(tasks[0], _task_output(), 0, tasks, [_task_output()])
                assert crew._replan_count == 1

                # Simulate second task completing with deviation
                crew._maybe_replan(tasks[1], _task_output(), 1, tasks, [_task_output()] * 2)
                assert crew._replan_count == 2

    def test_maybe_replan_stops_at_max_replans(self):
        crew, agents, tasks = self._setup_crew_with_planning(3)
        crew.max_replans = 1

        deviation = ReplanDecision(
            should_replan=True,
            reason="deviation",
            affected_task_numbers=[2],
        )

        revised = PlannerTaskPydanticOutput(
            list_of_plans_per_task=[
                PlanPerTask(task_number=1, task="T2", plan=" [NEW]"),
            ]
        )

        with patch.object(ReplanningEvaluator, "evaluate", return_value=deviation) as mock_eval:
            with patch.object(CrewPlanner, "_handle_crew_replanning", return_value=revised):
                crew._maybe_replan(tasks[0], _task_output(), 0, tasks, [_task_output()])
                assert crew._replan_count == 1

                # Second call should be skipped because max_replans=1
                crew._maybe_replan(tasks[1], _task_output(), 1, tasks, [_task_output()] * 2)
                assert crew._replan_count == 1  # unchanged
                # evaluate was only called once (the second time was short-circuited)
                assert mock_eval.call_count == 1


# ---------------------------------------------------------------------------
# Crew._handle_crew_planning stores original descriptions
# ---------------------------------------------------------------------------

class TestCrewPlanningStoresOriginals:
    def test_handle_crew_planning_stores_original_descriptions(self):
        agents = _make_agents(2)
        tasks = _make_tasks(agents)
        crew = Crew(agents=agents, tasks=tasks, planning=True)

        original_descs = [t.description for t in tasks]

        plans = [
            PlanPerTask(task_number=1, task="T1", plan=" [PLAN 1]"),
            PlanPerTask(task_number=2, task="T2", plan=" [PLAN 2]"),
        ]
        mock_result = PlannerTaskPydanticOutput(list_of_plans_per_task=plans)

        with patch.object(CrewPlanner, "_handle_crew_planning", return_value=mock_result):
            crew._handle_crew_planning()

        assert crew._original_task_descriptions == original_descs
        assert crew._replan_count == 0

    def test_handle_crew_planning_resets_replan_count(self):
        agents = _make_agents(1)
        tasks = _make_tasks(agents)
        crew = Crew(agents=agents, tasks=tasks, planning=True)
        crew._replan_count = 5  # leftover from previous execution

        plans = [PlanPerTask(task_number=1, task="T1", plan=" [PLAN]")]
        mock_result = PlannerTaskPydanticOutput(list_of_plans_per_task=plans)

        with patch.object(CrewPlanner, "_handle_crew_planning", return_value=mock_result):
            crew._handle_crew_planning()

        assert crew._replan_count == 0


# ---------------------------------------------------------------------------
# Sync execution integration test
# ---------------------------------------------------------------------------

class TestExecuteTasksWithReplanning:
    def test_execute_tasks_calls_maybe_replan_for_sync_tasks(self):
        """Verify that _maybe_replan is called after each sync task execution."""
        agents = _make_agents(2)
        tasks = _make_tasks(agents)
        crew = Crew(
            agents=agents,
            tasks=tasks,
            planning=True,
            replan_on_failure=True,
        )
        crew._original_task_descriptions = [t.description for t in tasks]
        crew._replan_count = 0

        output = _task_output("result")

        with patch.object(Task, "execute_sync", return_value=output):
            with patch.object(Crew, "_maybe_replan") as mock_replan:
                crew._execute_tasks(tasks)

        # Should be called once for each sync task
        assert mock_replan.call_count == 2

    def test_execute_tasks_without_replanning_is_unaffected(self):
        """Verify existing behaviour when replan_on_failure is False."""
        agents = _make_agents(2)
        tasks = _make_tasks(agents)
        crew = Crew(
            agents=agents,
            tasks=tasks,
            planning=False,
            replan_on_failure=False,
        )

        output = _task_output("result")

        with patch.object(Task, "execute_sync", return_value=output):
            with patch.object(Crew, "_maybe_replan") as mock_replan:
                result = crew._execute_tasks(tasks)

        # _maybe_replan is still called but returns immediately
        assert mock_replan.call_count == 2
        assert result is not None
