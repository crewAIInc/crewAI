"""Evaluates task results and triggers adaptive replanning when deviations are detected.

This module provides a lightweight evaluator that runs after each task completion
when adaptive replanning is enabled. It checks whether the task output deviates
significantly from what the plan assumed, and triggers replanning of remaining
tasks when necessary.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from crewai.agent import Agent
from crewai.llms.base_llm import BaseLLM
from crewai.task import Task
from crewai.tasks.task_output import TaskOutput


logger = logging.getLogger(__name__)


class ReplanDecision(BaseModel):
    """Structured decision from the replanning evaluator.

    Attributes:
        should_replan: Whether the remaining plan needs to be regenerated.
        reason: Human-readable explanation of why replanning is or isn't needed.
        deviation_score: A score from 0.0 to 1.0 indicating how much the result
            deviates from the plan's assumptions. 0.0 = perfect alignment,
            1.0 = completely off.
        affected_task_indices: 1-based indices of remaining tasks that are most
            affected by the deviation, if any.
    """

    should_replan: bool = Field(
        description="Whether the remaining tasks need to be replanned based on this result",
    )
    reason: str = Field(
        description="Explanation of why replanning is or is not needed",
    )
    deviation_score: float = Field(
        default=0.0,
        description="Score from 0.0 to 1.0 indicating deviation from plan assumptions",
        ge=0.0,
        le=1.0,
    )
    affected_task_indices: list[int] = Field(
        default_factory=list,
        description="1-based indices of remaining tasks affected by the deviation",
    )


class EvaluationCriteria(BaseModel):
    """Configurable criteria for the replanning evaluator.

    Attributes:
        quality_threshold: Minimum quality score (0-10) below which replanning
            is triggered. Defaults to 5.0.
        check_completeness: Whether to evaluate if the output is complete.
        check_relevance: Whether to evaluate if the output is relevant to the task.
        check_plan_alignment: Whether to evaluate if the output aligns with
            the original plan's assumptions.
        custom_criteria: Optional additional criteria as a free-text prompt
            appended to the evaluation query.
    """

    quality_threshold: float = Field(
        default=5.0,
        description="Minimum quality score (0-10) below which replanning is triggered",
        ge=0.0,
        le=10.0,
    )
    check_completeness: bool = Field(
        default=True,
        description="Whether to evaluate output completeness",
    )
    check_relevance: bool = Field(
        default=True,
        description="Whether to evaluate output relevance",
    )
    check_plan_alignment: bool = Field(
        default=True,
        description="Whether to evaluate alignment with the original plan",
    )
    custom_criteria: str | None = Field(
        default=None,
        description="Additional custom criteria for the evaluation",
    )


class ReplanningEvaluator:
    """Evaluates task results and decides whether replanning is needed.

    After each task completes, this evaluator makes a structured LLM call to
    determine whether the result deviates significantly from what the plan
    assumed. If it does, it returns a ReplanDecision indicating that replanning
    should occur.

    Attributes:
        llm: The LLM to use for evaluation. Defaults to a lightweight model.
        criteria: Configurable evaluation criteria.
    """

    def __init__(
        self,
        llm: str | BaseLLM | None = None,
        criteria: EvaluationCriteria | None = None,
    ) -> None:
        """Initialize the replanning evaluator.

        Args:
            llm: LLM to use for evaluation. Defaults to "gpt-4o-mini".
            criteria: Evaluation criteria. Defaults to sensible defaults.
        """
        self.llm = llm or "gpt-4o-mini"
        self.criteria = criteria or EvaluationCriteria()

    def evaluate(
        self,
        completed_task: Task,
        task_output: TaskOutput,
        original_plan: str,
        remaining_tasks: list[Task],
        completed_outputs: list[TaskOutput],
    ) -> ReplanDecision:
        """Evaluate whether a task result requires replanning.

        Makes a structured LLM call to assess whether the task output deviates
        from the plan's assumptions in a way that would affect remaining tasks.

        Args:
            completed_task: The task that just completed.
            task_output: The output of the completed task.
            original_plan: The original plan text for context.
            remaining_tasks: Tasks that have not yet been executed.
            completed_outputs: Outputs from previously completed tasks.

        Returns:
            A ReplanDecision indicating whether replanning is needed.
        """
        if not remaining_tasks:
            return ReplanDecision(
                should_replan=False,
                reason="No remaining tasks to replan.",
                deviation_score=0.0,
            )

        evaluator_agent = self._create_evaluator_agent()
        evaluation_task = self._create_evaluation_task(
            evaluator_agent=evaluator_agent,
            completed_task=completed_task,
            task_output=task_output,
            original_plan=original_plan,
            remaining_tasks=remaining_tasks,
            completed_outputs=completed_outputs,
        )

        result = evaluation_task.execute_sync()

        if isinstance(result.pydantic, ReplanDecision):
            return result.pydantic

        # Fallback: if structured output fails, don't replan
        logger.warning(
            "Replanning evaluator could not parse structured output. "
            "Continuing without replanning."
        )
        return ReplanDecision(
            should_replan=False,
            reason="Evaluator output could not be parsed; defaulting to no replan.",
            deviation_score=0.0,
        )

    def _create_evaluator_agent(self) -> Agent:
        """Create the lightweight evaluator agent.

        Returns:
            An Agent configured for evaluating task deviations.
        """
        return Agent(
            role="Task Result Evaluator",
            goal=(
                "Evaluate whether a task's output deviates from the original plan's "
                "assumptions in a way that requires the remaining tasks to be replanned."
            ),
            backstory=(
                "You are an expert evaluator that assesses task outputs against "
                "planned expectations. You identify when results contradict assumptions, "
                "when data is missing or unexpected, or when the planned approach has "
                "become infeasible based on the actual results."
            ),
            llm=self.llm,
            verbose=False,
        )

    def _create_evaluation_task(
        self,
        evaluator_agent: Agent,
        completed_task: Task,
        task_output: TaskOutput,
        original_plan: str,
        remaining_tasks: list[Task],
        completed_outputs: list[TaskOutput],
    ) -> Task:
        """Create the evaluation task with all relevant context.

        Args:
            evaluator_agent: The agent to perform the evaluation.
            completed_task: The task that was just completed.
            task_output: The output of the completed task.
            original_plan: The original crew plan.
            remaining_tasks: Tasks that still need to be executed.
            completed_outputs: Outputs from previously completed tasks.

        Returns:
            A Task configured for evaluating the need for replanning.
        """
        criteria_text = self._build_criteria_text()
        remaining_summary = "\n".join(
            f"  - Task {i + 1}: {t.description}" for i, t in enumerate(remaining_tasks)
        )
        completed_summary = "\n".join(
            f"  - {o.description}: {o.raw[:200]}..." if len(o.raw) > 200 else f"  - {o.description}: {o.raw}"
            for o in completed_outputs
        )

        description = (
            f"Evaluate whether the following task result requires replanning.\n\n"
            f"ORIGINAL PLAN:\n{original_plan}\n\n"
            f"COMPLETED TASK:\n"
            f"  Description: {completed_task.description}\n"
            f"  Expected Output: {completed_task.expected_output}\n\n"
            f"ACTUAL RESULT:\n{task_output.raw}\n\n"
            f"PREVIOUSLY COMPLETED TASKS:\n{completed_summary or '  (none)'}\n\n"
            f"REMAINING TASKS:\n{remaining_summary}\n\n"
            f"EVALUATION CRITERIA:\n{criteria_text}\n\n"
            f"Based on the above, determine whether the remaining tasks need to be "
            f"replanned. Consider whether the actual result contradicts the plan's "
            f"assumptions, whether expected data is missing, whether the approach "
            f"is still feasible, and whether the remaining tasks can still produce "
            f"good results with the current plan."
        )

        return Task(
            description=description,
            expected_output=(
                "A structured evaluation indicating whether replanning is needed, "
                "with a reason, deviation score, and affected task indices."
            ),
            agent=evaluator_agent,
            output_pydantic=ReplanDecision,
        )

    def _build_criteria_text(self) -> str:
        """Build human-readable evaluation criteria text.

        Returns:
            A string describing the evaluation criteria.
        """
        criteria_parts = []
        if self.criteria.check_completeness:
            criteria_parts.append(
                "- COMPLETENESS: Is the output complete? Does it contain all "
                "the expected information?"
            )
        if self.criteria.check_relevance:
            criteria_parts.append(
                "- RELEVANCE: Is the output relevant to the task description "
                "and expected output?"
            )
        if self.criteria.check_plan_alignment:
            criteria_parts.append(
                "- PLAN ALIGNMENT: Does the output align with what the plan "
                "assumed would happen? Are the plan's assumptions still valid?"
            )
        criteria_parts.append(
            f"- QUALITY THRESHOLD: Results scoring below "
            f"{self.criteria.quality_threshold}/10 on quality should trigger replanning."
        )
        if self.criteria.custom_criteria:
            criteria_parts.append(f"- CUSTOM: {self.criteria.custom_criteria}")

        return "\n".join(criteria_parts)
