"""Evaluates whether task results deviate from the original plan and triggers replanning."""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from crewai.agent import Agent
from crewai.llms.base_llm import BaseLLM
from crewai.task import Task
from crewai.tasks.task_output import TaskOutput


logger = logging.getLogger(__name__)


class ReplanDecision(BaseModel):
    """Structured decision on whether replanning is needed.

    Attributes:
        should_replan: Whether the crew should generate a new plan for remaining tasks.
        reason: Explanation of why replanning is or is not needed.
        affected_task_numbers: 1-indexed task numbers of remaining tasks most affected
            by the deviation. Empty if should_replan is False.
    """

    should_replan: bool = Field(
        description="Whether the task result deviates significantly from the plan, requiring replanning.",
    )
    reason: str = Field(
        description="A concise explanation of why replanning is or is not needed.",
    )
    affected_task_numbers: list[int] = Field(
        default_factory=list,
        description="1-indexed task numbers of remaining tasks most affected by the deviation.",
    )


class ReplanningEvaluator:
    """Evaluates task outputs to decide if the crew's plan needs to be revised.

    After each task completes, this evaluator makes a lightweight LLM call to
    determine whether the result deviates significantly from what the plan
    assumed. If so, it signals that replanning should occur.

    Example usage::

        evaluator = ReplanningEvaluator(llm="gpt-4o-mini")
        decision = evaluator.evaluate(
            completed_task=task,
            task_output=output,
            original_plan="Step 1: ...",
            remaining_tasks=remaining,
        )
        if decision.should_replan:
            # trigger replanning for remaining tasks
            ...

    Args:
        llm: The language model to use for evaluation. Accepts a string model
            name or a BaseLLM instance. Defaults to ``"gpt-4o-mini"``.
    """

    def __init__(self, llm: str | BaseLLM | None = None) -> None:
        self.llm = llm or "gpt-4o-mini"

    def evaluate(
        self,
        completed_task: Task,
        task_output: TaskOutput,
        original_plan: str,
        remaining_tasks: list[Task],
    ) -> ReplanDecision:
        """Evaluate whether a task result deviates from the plan.

        Args:
            completed_task: The task that just finished executing.
            task_output: The output produced by the completed task.
            original_plan: The plan text that was appended to the task description.
            remaining_tasks: Tasks that have not yet been executed.

        Returns:
            A ReplanDecision indicating whether replanning is needed.
        """
        evaluation_agent = Agent(
            role="Replanning Evaluator",
            goal=(
                "Evaluate whether the result of a completed task deviates "
                "significantly from what the plan assumed, and determine if "
                "the remaining tasks need a revised plan."
            ),
            backstory=(
                "You are an expert at evaluating execution plans. You compare "
                "actual task results against planned expectations and identify "
                "deviations that would make the remaining plan ineffective."
            ),
            llm=self.llm,
        )

        remaining_summary = self._summarize_remaining_tasks(remaining_tasks)

        evaluation_task = Task(
            description=(
                "Evaluate whether the following task result deviates significantly "
                "from what the original plan assumed.\n\n"
                f"## Completed Task\n"
                f"Description: {completed_task.description}\n"
                f"Expected Output: {completed_task.expected_output}\n\n"
                f"## Plan for this Task\n{original_plan}\n\n"
                f"## Actual Result\n{task_output.raw}\n\n"
                f"## Remaining Tasks\n{remaining_summary}\n\n"
                "Based on the above, decide if the actual result deviates enough "
                "from the plan's assumptions that the remaining tasks need replanning. "
                "Minor differences or format changes do NOT require replanning. "
                "Only significant deviations (missing data, errors, completely "
                "different approach needed, infeasible assumptions) should trigger replanning."
            ),
            expected_output=(
                "A structured decision indicating whether replanning is needed, "
                "with a reason and the affected task numbers."
            ),
            agent=evaluation_agent,
            output_pydantic=ReplanDecision,
        )

        result = evaluation_task.execute_sync()

        if isinstance(result.pydantic, ReplanDecision):
            return result.pydantic

        logger.warning(
            "Failed to get structured ReplanDecision, defaulting to no replan"
        )
        return ReplanDecision(
            should_replan=False,
            reason="Failed to evaluate task output against plan.",
            affected_task_numbers=[],
        )

    @staticmethod
    def _summarize_remaining_tasks(remaining_tasks: list[Task]) -> str:
        """Create a summary of remaining tasks for evaluation context.

        Args:
            remaining_tasks: Tasks that have not yet been executed.

        Returns:
            A formatted string summarizing the remaining tasks.
        """
        if not remaining_tasks:
            return "No remaining tasks."

        summaries = []
        for idx, task in enumerate(remaining_tasks, start=1):
            agent_role = task.agent.role if task.agent else "Unassigned"
            summaries.append(
                f"Task {idx}: {task.description}\n"
                f"  Expected Output: {task.expected_output}\n"
                f"  Agent: {agent_role}"
            )
        return "\n".join(summaries)
