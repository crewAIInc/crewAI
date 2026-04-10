"""Tests for the adaptive replanning evaluator module.

Tests cover the ReplanningEvaluator, ReplanDecision, EvaluationCriteria,
and the integration of replanning into Crew execution.
"""

from unittest.mock import MagicMock, patch, PropertyMock

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
    EvaluationCriteria,
    ReplanDecision,
    ReplanningEvaluator,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def simple_agent():
    """Create a basic agent for testing."""
    return Agent(role="Researcher", goal="Research topics", backstory="Expert researcher")


@pytest.fixture
def simple_tasks(simple_agent):
    """Create a list of three simple tasks."""
    return [
        Task(
            description="Research the topic",
            expected_output="Research findings",
            agent=simple_agent,
        ),
        Task(
            description="Analyze the research",
            expected_output="Analysis report",
            agent=simple_agent,
        ),
        Task(
            description="Write the final report",
            expected_output="Final report",
            agent=simple_agent,
        ),
    ]


@pytest.fixture
def task_output_success():
    """A successful task output."""
    return TaskOutput(
        description="Research the topic",
        raw="Found extensive data on the topic with 10 relevant sources.",
        agent="Researcher",
    )


@pytest.fixture
def task_output_failure():
    """A task output that deviates from the plan."""
    return TaskOutput(
        description="Research the topic",
        raw="No data found. The topic does not exist in any known database.",
        agent="Researcher",
    )


# ── ReplanDecision Model Tests ───────────────────────────────────────────────


class TestReplanDecision:
    def test_replan_decision_defaults(self):
        decision = ReplanDecision(
            should_replan=False,
            reason="All good",
        )
        assert decision.should_replan is False
        assert decision.reason == "All good"
        assert decision.deviation_score == 0.0
        assert decision.affected_task_indices == []

    def test_replan_decision_with_all_fields(self):
        decision = ReplanDecision(
            should_replan=True,
            reason="Data not found",
            deviation_score=0.8,
            affected_task_indices=[2, 3],
        )
        assert decision.should_replan is True
        assert decision.deviation_score == 0.8
        assert decision.affected_task_indices == [2, 3]

    def test_replan_decision_deviation_score_bounds(self):
        """Deviation score must be between 0.0 and 1.0."""
        with pytest.raises(Exception):
            ReplanDecision(
                should_replan=False,
                reason="test",
                deviation_score=1.5,
            )

    def test_replan_decision_negative_deviation_score(self):
        with pytest.raises(Exception):
            ReplanDecision(
                should_replan=False,
                reason="test",
                deviation_score=-0.1,
            )


# ── EvaluationCriteria Model Tests ───────────────────────────────────────────


class TestEvaluationCriteria:
    def test_default_criteria(self):
        criteria = EvaluationCriteria()
        assert criteria.quality_threshold == 5.0
        assert criteria.check_completeness is True
        assert criteria.check_relevance is True
        assert criteria.check_plan_alignment is True
        assert criteria.custom_criteria is None

    def test_custom_criteria(self):
        criteria = EvaluationCriteria(
            quality_threshold=7.0,
            check_completeness=False,
            custom_criteria="Must include at least 3 data sources",
        )
        assert criteria.quality_threshold == 7.0
        assert criteria.check_completeness is False
        assert criteria.custom_criteria == "Must include at least 3 data sources"

    def test_quality_threshold_bounds(self):
        with pytest.raises(Exception):
            EvaluationCriteria(quality_threshold=11.0)
        with pytest.raises(Exception):
            EvaluationCriteria(quality_threshold=-1.0)


# ── ReplanningEvaluator Tests ────────────────────────────────────────────────


class TestReplanningEvaluator:
    def test_default_initialization(self):
        evaluator = ReplanningEvaluator()
        assert evaluator.llm == "gpt-4o-mini"
        assert isinstance(evaluator.criteria, EvaluationCriteria)

    def test_custom_initialization(self):
        criteria = EvaluationCriteria(quality_threshold=8.0)
        evaluator = ReplanningEvaluator(llm="gpt-4o", criteria=criteria)
        assert evaluator.llm == "gpt-4o"
        assert evaluator.criteria.quality_threshold == 8.0

    def test_no_remaining_tasks_returns_no_replan(
        self, simple_tasks, task_output_success
    ):
        evaluator = ReplanningEvaluator()
        decision = evaluator.evaluate(
            completed_task=simple_tasks[0],
            task_output=task_output_success,
            original_plan="Plan text",
            remaining_tasks=[],
            completed_outputs=[task_output_success],
        )
        assert decision.should_replan is False
        assert "No remaining tasks" in decision.reason

    def test_evaluate_returns_replan_decision(
        self, simple_tasks, task_output_failure
    ):
        """When the LLM evaluator says to replan, the decision reflects that."""
        evaluator = ReplanningEvaluator()
        mock_decision = ReplanDecision(
            should_replan=True,
            reason="Research found no data, planned analysis is infeasible",
            deviation_score=0.9,
            affected_task_indices=[2, 3],
        )

        mock_result = MagicMock()
        mock_result.pydantic = mock_decision

        with patch.object(Task, "execute_sync", return_value=mock_result):
            decision = evaluator.evaluate(
                completed_task=simple_tasks[0],
                task_output=task_output_failure,
                original_plan="Task 1: Research data\nTask 2: Analyze data\nTask 3: Write report",
                remaining_tasks=simple_tasks[1:],
                completed_outputs=[task_output_failure],
            )
        assert decision.should_replan is True
        assert "infeasible" in decision.reason

    def test_evaluate_returns_no_replan(
        self, simple_tasks, task_output_success
    ):
        """When the LLM evaluator says no replan needed."""
        evaluator = ReplanningEvaluator()
        mock_decision = ReplanDecision(
            should_replan=False,
            reason="Results align with plan assumptions",
            deviation_score=0.1,
        )

        mock_result = MagicMock()
        mock_result.pydantic = mock_decision

        with patch.object(Task, "execute_sync", return_value=mock_result):
            decision = evaluator.evaluate(
                completed_task=simple_tasks[0],
                task_output=task_output_success,
                original_plan="Task 1: Research data",
                remaining_tasks=simple_tasks[1:],
                completed_outputs=[task_output_success],
            )
        assert decision.should_replan is False

    def test_evaluate_fallback_on_parse_failure(
        self, simple_tasks, task_output_success
    ):
        """When structured output parsing fails, default to no replan."""
        evaluator = ReplanningEvaluator()

        mock_result = MagicMock()
        mock_result.pydantic = "not a ReplanDecision"

        with patch.object(Task, "execute_sync", return_value=mock_result):
            decision = evaluator.evaluate(
                completed_task=simple_tasks[0],
                task_output=task_output_success,
                original_plan="Plan",
                remaining_tasks=simple_tasks[1:],
                completed_outputs=[task_output_success],
            )
        assert decision.should_replan is False
        assert "could not be parsed" in decision.reason

    def test_build_criteria_text_all_enabled(self):
        evaluator = ReplanningEvaluator(
            criteria=EvaluationCriteria(
                custom_criteria="Must include citations",
            )
        )
        text = evaluator._build_criteria_text()
        assert "COMPLETENESS" in text
        assert "RELEVANCE" in text
        assert "PLAN ALIGNMENT" in text
        assert "QUALITY THRESHOLD" in text
        assert "Must include citations" in text

    def test_build_criteria_text_selective(self):
        evaluator = ReplanningEvaluator(
            criteria=EvaluationCriteria(
                check_completeness=False,
                check_relevance=False,
            )
        )
        text = evaluator._build_criteria_text()
        assert "COMPLETENESS" not in text
        assert "RELEVANCE" not in text
        assert "PLAN ALIGNMENT" in text

    def test_evaluator_agent_creation(self):
        evaluator = ReplanningEvaluator()
        agent = evaluator._create_evaluator_agent()
        assert agent.role == "Task Result Evaluator"
        assert "deviates" in agent.goal


# ── CrewPlanner.replan() Tests ───────────────────────────────────────────────


class TestCrewPlannerReplan:
    def test_replan_returns_revised_plans(self, simple_tasks):
        planner = CrewPlanner(tasks=simple_tasks, planning_agent_llm=None)

        mock_output = PlannerTaskPydanticOutput(
            list_of_plans_per_task=[
                PlanPerTask(
                    task_number=1,
                    task="Analyze the research",
                    plan="Revised: Use alternative data sources",
                ),
                PlanPerTask(
                    task_number=2,
                    task="Write the final report",
                    plan="Revised: Focus on available data only",
                ),
            ]
        )

        mock_result = MagicMock()
        mock_result.pydantic = mock_output

        completed_output = TaskOutput(
            description="Research",
            raw="No data found",
            agent="Researcher",
        )

        with patch.object(Task, "execute_sync", return_value=mock_result):
            result = planner.replan(
                completed_results=[completed_output],
                remaining_tasks=simple_tasks[1:],
                deviation_reason="No data found during research",
            )

        assert len(result.list_of_plans_per_task) == 2
        assert "alternative data sources" in result.list_of_plans_per_task[0].plan

    def test_replan_raises_on_failure(self, simple_tasks):
        planner = CrewPlanner(tasks=simple_tasks, planning_agent_llm=None)

        mock_result = MagicMock()
        mock_result.pydantic = "not valid"

        completed_output = TaskOutput(
            description="Research", raw="No data", agent="Researcher"
        )

        with patch.object(Task, "execute_sync", return_value=mock_result):
            with pytest.raises(ValueError, match="Failed to get the Replanning output"):
                planner.replan(
                    completed_results=[completed_output],
                    remaining_tasks=simple_tasks[1:],
                    deviation_reason="test",
                )

    def test_create_remaining_tasks_summary(self, simple_tasks):
        summary = CrewPlanner._create_remaining_tasks_summary(simple_tasks[:2])
        assert "Task 1:" in summary
        assert "Task 2:" in summary
        assert "Research the topic" in summary


# ── Crew Integration Tests ───────────────────────────────────────────────────


class TestCrewReplanningIntegration:
    def test_crew_replan_fields_defaults(self, simple_agent, simple_tasks):
        crew = Crew(
            agents=[simple_agent],
            tasks=simple_tasks,
        )
        assert crew.replan_on_failure is False
        assert crew.max_replans == 3
        assert crew.replanning_evaluator is None
        assert crew.evaluation_criteria is None

    def test_crew_replan_fields_custom(self, simple_agent, simple_tasks):
        criteria = EvaluationCriteria(quality_threshold=8.0)
        evaluator = ReplanningEvaluator(criteria=criteria)

        crew = Crew(
            agents=[simple_agent],
            tasks=simple_tasks,
            planning=True,
            replan_on_failure=True,
            max_replans=5,
            replanning_evaluator=evaluator,
            evaluation_criteria=criteria,
        )
        assert crew.replan_on_failure is True
        assert crew.max_replans == 5
        assert crew.replanning_evaluator is evaluator
        assert crew.evaluation_criteria.quality_threshold == 8.0

    def test_should_evaluate_for_replan_true(self, simple_agent, simple_tasks):
        crew = Crew(
            agents=[simple_agent],
            tasks=simple_tasks,
            planning=True,
            replan_on_failure=True,
            max_replans=3,
        )
        crew._replan_count = 0
        assert crew._should_evaluate_for_replan() is True

    def test_should_evaluate_for_replan_false_no_planning(
        self, simple_agent, simple_tasks
    ):
        crew = Crew(
            agents=[simple_agent],
            tasks=simple_tasks,
            planning=False,
            replan_on_failure=True,
        )
        assert crew._should_evaluate_for_replan() is False

    def test_should_evaluate_for_replan_false_not_enabled(
        self, simple_agent, simple_tasks
    ):
        crew = Crew(
            agents=[simple_agent],
            tasks=simple_tasks,
            planning=True,
            replan_on_failure=False,
        )
        assert crew._should_evaluate_for_replan() is False

    def test_should_evaluate_max_replans_reached(
        self, simple_agent, simple_tasks
    ):
        crew = Crew(
            agents=[simple_agent],
            tasks=simple_tasks,
            planning=True,
            replan_on_failure=True,
            max_replans=2,
        )
        crew._replan_count = 2
        assert crew._should_evaluate_for_replan() is False

    def test_evaluate_and_replan_triggers_replanning(
        self, simple_agent, simple_tasks
    ):
        """When evaluator says replan, the plan should be revised."""
        crew = Crew(
            agents=[simple_agent],
            tasks=simple_tasks,
            planning=True,
            replan_on_failure=True,
            max_replans=3,
        )
        crew._original_plan_text = "Task 1: Research\nTask 2: Analyze\nTask 3: Report"
        crew._replan_count = 0

        mock_decision = ReplanDecision(
            should_replan=True,
            reason="No data available",
            deviation_score=0.9,
        )
        mock_evaluator = MagicMock(spec=ReplanningEvaluator)
        mock_evaluator.evaluate.return_value = mock_decision
        crew.replanning_evaluator = mock_evaluator

        mock_revised_plan = PlannerTaskPydanticOutput(
            list_of_plans_per_task=[
                PlanPerTask(task_number=1, task="Analyze", plan="Use alternative sources"),
                PlanPerTask(task_number=2, task="Report", plan="Report on available data"),
            ]
        )
        task_output = TaskOutput(
            description="Research", raw="No data", agent="Researcher"
        )

        with patch.object(CrewPlanner, "replan", return_value=mock_revised_plan):
            crew._evaluate_and_replan(
                completed_task=simple_tasks[0],
                task_output=task_output,
                task_outputs=[task_output],
                remaining_tasks=simple_tasks[1:],
            )

        assert crew._replan_count == 1
        assert "[REVISED PLAN]" in simple_tasks[1].description

    def test_evaluate_and_replan_no_replan_needed(
        self, simple_agent, simple_tasks
    ):
        crew = Crew(
            agents=[simple_agent],
            tasks=simple_tasks,
            planning=True,
            replan_on_failure=True,
        )
        crew._original_plan_text = "Plan text"
        crew._replan_count = 0

        mock_decision = ReplanDecision(
            should_replan=False,
            reason="Results align with plan",
        )
        mock_evaluator = MagicMock(spec=ReplanningEvaluator)
        mock_evaluator.evaluate.return_value = mock_decision
        crew.replanning_evaluator = mock_evaluator

        task_output = TaskOutput(
            description="Research", raw="Good data found", agent="Researcher"
        )

        original_desc = simple_tasks[1].description
        crew._evaluate_and_replan(
            completed_task=simple_tasks[0],
            task_output=task_output,
            task_outputs=[task_output],
            remaining_tasks=simple_tasks[1:],
        )

        assert crew._replan_count == 0
        assert simple_tasks[1].description == original_desc

    def test_evaluate_and_replan_handles_evaluation_error(
        self, simple_agent, simple_tasks
    ):
        """Evaluation errors should not crash execution."""
        crew = Crew(
            agents=[simple_agent],
            tasks=simple_tasks,
            planning=True,
            replan_on_failure=True,
        )
        crew._original_plan_text = "Plan text"
        crew._replan_count = 0

        mock_evaluator = MagicMock(spec=ReplanningEvaluator)
        mock_evaluator.evaluate.side_effect = RuntimeError("LLM unavailable")
        crew.replanning_evaluator = mock_evaluator

        task_output = TaskOutput(
            description="Research", raw="data", agent="Researcher"
        )

        # Should not raise
        crew._evaluate_and_replan(
            completed_task=simple_tasks[0],
            task_output=task_output,
            task_outputs=[task_output],
            remaining_tasks=simple_tasks[1:],
        )
        assert crew._replan_count == 0

    def test_evaluate_and_replan_handles_replanning_error(
        self, simple_agent, simple_tasks
    ):
        """Replanning errors should not crash execution."""
        crew = Crew(
            agents=[simple_agent],
            tasks=simple_tasks,
            planning=True,
            replan_on_failure=True,
        )
        crew._original_plan_text = "Plan"
        crew._replan_count = 0

        mock_decision = ReplanDecision(
            should_replan=True, reason="Need replan"
        )
        mock_evaluator = MagicMock(spec=ReplanningEvaluator)
        mock_evaluator.evaluate.return_value = mock_decision
        crew.replanning_evaluator = mock_evaluator

        task_output = TaskOutput(
            description="Research", raw="data", agent="Researcher"
        )

        with patch.object(CrewPlanner, "replan", side_effect=ValueError("Failed")):
            crew._evaluate_and_replan(
                completed_task=simple_tasks[0],
                task_output=task_output,
                task_outputs=[task_output],
                remaining_tasks=simple_tasks[1:],
            )

        # Replan count increments but doesn't crash
        assert crew._replan_count == 1

    def test_replan_count_increments(self, simple_agent, simple_tasks):
        """Each successful replan trigger increments the counter."""
        crew = Crew(
            agents=[simple_agent],
            tasks=simple_tasks,
            planning=True,
            replan_on_failure=True,
            max_replans=5,
        )
        crew._original_plan_text = "Plan"
        crew._replan_count = 0

        mock_decision = ReplanDecision(
            should_replan=True, reason="Need replan"
        )
        mock_evaluator = MagicMock(spec=ReplanningEvaluator)
        mock_evaluator.evaluate.return_value = mock_decision
        crew.replanning_evaluator = mock_evaluator

        mock_plan = PlannerTaskPydanticOutput(
            list_of_plans_per_task=[
                PlanPerTask(task_number=1, task="t", plan="revised plan"),
            ]
        )

        task_output = TaskOutput(
            description="Task", raw="output", agent="Agent"
        )

        with patch.object(CrewPlanner, "replan", return_value=mock_plan):
            for i in range(3):
                crew._evaluate_and_replan(
                    completed_task=simple_tasks[0],
                    task_output=task_output,
                    task_outputs=[task_output],
                    remaining_tasks=simple_tasks[1:2],
                )

        assert crew._replan_count == 3

    def test_revised_plan_strips_old_revision(self, simple_agent, simple_tasks):
        """Multiple replans should not stack [REVISED PLAN] sections."""
        crew = Crew(
            agents=[simple_agent],
            tasks=simple_tasks,
            planning=True,
            replan_on_failure=True,
        )
        crew._original_plan_text = "Plan"
        crew._replan_count = 0

        mock_decision = ReplanDecision(
            should_replan=True, reason="Need replan"
        )
        mock_evaluator = MagicMock(spec=ReplanningEvaluator)
        mock_evaluator.evaluate.return_value = mock_decision
        crew.replanning_evaluator = mock_evaluator

        # First replan
        mock_plan_1 = PlannerTaskPydanticOutput(
            list_of_plans_per_task=[
                PlanPerTask(task_number=1, task="Analyze", plan="Plan v1"),
            ]
        )
        task_output = TaskOutput(
            description="Research", raw="data", agent="Agent"
        )

        remaining = simple_tasks[1:2]

        with patch.object(CrewPlanner, "replan", return_value=mock_plan_1):
            crew._evaluate_and_replan(
                completed_task=simple_tasks[0],
                task_output=task_output,
                task_outputs=[task_output],
                remaining_tasks=remaining,
            )

        assert remaining[0].description.count("[REVISED PLAN]") == 1

        # Second replan - should replace, not stack
        mock_plan_2 = PlannerTaskPydanticOutput(
            list_of_plans_per_task=[
                PlanPerTask(task_number=1, task="Analyze", plan="Plan v2"),
            ]
        )

        with patch.object(CrewPlanner, "replan", return_value=mock_plan_2):
            crew._evaluate_and_replan(
                completed_task=simple_tasks[0],
                task_output=task_output,
                task_outputs=[task_output],
                remaining_tasks=remaining,
            )

        assert remaining[0].description.count("[REVISED PLAN]") == 1
        assert "Plan v2" in remaining[0].description

    def test_get_replanning_evaluator_default(self, simple_agent, simple_tasks):
        crew = Crew(
            agents=[simple_agent],
            tasks=simple_tasks,
            planning=True,
            replan_on_failure=True,
        )
        evaluator = crew._get_replanning_evaluator()
        assert isinstance(evaluator, ReplanningEvaluator)

    def test_get_replanning_evaluator_custom(self, simple_agent, simple_tasks):
        custom_evaluator = ReplanningEvaluator(llm="gpt-4o")
        crew = Crew(
            agents=[simple_agent],
            tasks=simple_tasks,
            planning=True,
            replan_on_failure=True,
            replanning_evaluator=custom_evaluator,
        )
        assert crew._get_replanning_evaluator() is custom_evaluator

    def test_backwards_compatibility_default_crew(self, simple_agent, simple_tasks):
        """Existing crews without replan_on_failure are unaffected."""
        crew = Crew(
            agents=[simple_agent],
            tasks=simple_tasks,
            planning=True,
        )
        assert crew.replan_on_failure is False
        assert crew._should_evaluate_for_replan() is False
