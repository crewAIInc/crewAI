"""PlannerObserver: Observation phase after each step execution.

Implements the "Observe" phase. After every step execution, the Planner
analyzes what happened, what new information was learned, and whether the
remaining plan is still valid.

This is NOT an error detector — it runs on every step, including successes,
to incorporate runtime observations into the remaining plan.

Refinements are structured (StepRefinement objects) and applied directly
from the observation result — no second LLM call required.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.observation_events import (
    StepObservationCompletedEvent,
    StepObservationFailedEvent,
    StepObservationStartedEvent,
)
from crewai.utilities.i18n import I18N, get_i18n
from crewai.utilities.llm_utils import create_llm
from crewai.utilities.planning_types import StepObservation, TodoItem
from crewai.utilities.types import LLMMessage


if TYPE_CHECKING:
    from crewai.agent import Agent
    from crewai.task import Task

logger = logging.getLogger(__name__)


class PlannerObserver:
    """Observes step execution results and decides on plan continuation.

    After EVERY step execution, this class:
    1. Analyzes what the step accomplished
    2. Identifies new information learned
    3. Decides if the remaining plan is still valid
    4. Suggests lightweight refinements or triggers full replanning

    LLM resolution (magical fallback):
    - If ``agent.planning_config.llm`` is explicitly set → use that
    - Otherwise → fall back to ``agent.llm`` (same LLM for everything)

    Args:
        agent: The agent instance (for LLM resolution and config).
        task: Optional task context (for description and expected output).
    """

    def __init__(self, agent: Agent, task: Task | None = None) -> None:
        self.agent = agent
        self.task = task
        self.llm = self._resolve_llm()
        self._i18n: I18N = get_i18n()

    def _resolve_llm(self) -> Any:
        """Resolve which LLM to use for observation/planning.

        Mirrors AgentReasoning._resolve_llm(): uses planning_config.llm
        if explicitly set, otherwise falls back to agent.llm.

        Returns:
            The resolved LLM instance.
        """
        from crewai.llm import LLM

        config = getattr(self.agent, "planning_config", None)
        if config is not None and config.llm is not None:
            if isinstance(config.llm, LLM):
                return config.llm
            return create_llm(config.llm)
        return self.agent.llm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def observe(
        self,
        completed_step: TodoItem,
        result: str,
        all_completed: list[TodoItem],
        remaining_todos: list[TodoItem],
    ) -> StepObservation:
        """Observe a step's result and decide on plan continuation.

        This runs after EVERY step execution — not just failures.

        Args:
            completed_step: The todo item that was just executed.
            result: The final result string from the step.
            all_completed: All previously completed todos (for context).
            remaining_todos: The pending todos still in the plan.

        Returns:
            StepObservation with the Planner's analysis. Any suggested
            refinements are structured StepRefinement objects ready for
            direct application — no second LLM call needed.
        """
        agent_role = self.agent.role

        crewai_event_bus.emit(
            self.agent,
            event=StepObservationStartedEvent(
                agent_role=agent_role,
                step_number=completed_step.step_number,
                step_description=completed_step.description,
                from_task=self.task,
                from_agent=self.agent,
            ),
        )

        messages = self._build_observation_messages(
            completed_step, result, all_completed, remaining_todos
        )

        try:
            response = self.llm.call(
                messages,
                response_model=StepObservation,
                from_task=self.task,
                from_agent=self.agent,
            )

            if isinstance(response, StepObservation):
                observation = response
            else:
                observation = StepObservation(
                    step_completed_successfully=True,
                    key_information_learned=str(response) if response else "",
                    remaining_plan_still_valid=True,
                )

            refinement_summaries = (
                [
                    f"Step {r.step_number}: {r.new_description}"
                    for r in observation.suggested_refinements
                ]
                if observation.suggested_refinements
                else None
            )

            crewai_event_bus.emit(
                self.agent,
                event=StepObservationCompletedEvent(
                    agent_role=agent_role,
                    step_number=completed_step.step_number,
                    step_description=completed_step.description,
                    step_completed_successfully=observation.step_completed_successfully,
                    key_information_learned=observation.key_information_learned,
                    remaining_plan_still_valid=observation.remaining_plan_still_valid,
                    needs_full_replan=observation.needs_full_replan,
                    replan_reason=observation.replan_reason,
                    goal_already_achieved=observation.goal_already_achieved,
                    suggested_refinements=refinement_summaries,
                    from_task=self.task,
                    from_agent=self.agent,
                ),
            )

            return observation

        except Exception as e:
            logger.warning(
                f"Observation LLM call failed: {e}. Defaulting to conservative replan."
            )

            crewai_event_bus.emit(
                self.agent,
                event=StepObservationFailedEvent(
                    agent_role=agent_role,
                    step_number=completed_step.step_number,
                    step_description=completed_step.description,
                    error=str(e),
                    from_task=self.task,
                    from_agent=self.agent,
                ),
            )

            return StepObservation(
                step_completed_successfully=False,
                key_information_learned="",
                remaining_plan_still_valid=False,
                needs_full_replan=True,
                replan_reason="Observer failed to evaluate step result safely",
            )

    def apply_refinements(
        self,
        observation: StepObservation,
        remaining_todos: list[TodoItem],
    ) -> list[TodoItem]:
        """Apply structured refinements from the observation directly to todo descriptions.

        No LLM call needed — refinements are already structured StepRefinement
        objects produced by the observation call. This is a pure in-memory update.

        Args:
            observation: The observation containing structured refinements.
            remaining_todos: The pending todos to update in-place.

        Returns:
            The same todo list with updated descriptions where refinements applied.
        """
        if not observation.suggested_refinements:
            return remaining_todos

        todo_by_step: dict[int, TodoItem] = {t.step_number: t for t in remaining_todos}
        for refinement in observation.suggested_refinements:
            if refinement.step_number in todo_by_step and refinement.new_description:
                todo_by_step[refinement.step_number].description = refinement.new_description

        return remaining_todos

    # ------------------------------------------------------------------
    # Internal: Message building
    # ------------------------------------------------------------------

    def _build_observation_messages(
        self,
        completed_step: TodoItem,
        result: str,
        all_completed: list[TodoItem],
        remaining_todos: list[TodoItem],
    ) -> list[LLMMessage]:
        """Build messages for the observation LLM call."""
        task_desc = ""
        task_goal = ""
        if self.task:
            task_desc = self.task.description or ""
            task_goal = self.task.expected_output or ""

        system_prompt = self._i18n.retrieve("planning", "observation_system_prompt")

        # Build context of what's been done
        completed_summary = ""
        if all_completed:
            completed_lines = []
            for todo in all_completed:
                result_preview = (todo.result or "")[:200]
                completed_lines.append(
                    f"  Step {todo.step_number}: {todo.description}\n"
                    f"    Result: {result_preview}"
                )
            completed_summary = "\n## Previously completed steps:\n" + "\n".join(
                completed_lines
            )

        # Build remaining plan
        remaining_summary = ""
        if remaining_todos:
            remaining_lines = [
                f"  Step {todo.step_number}: {todo.description}"
                for todo in remaining_todos
            ]
            remaining_summary = "\n## Remaining plan steps:\n" + "\n".join(
                remaining_lines
            )

        user_prompt = self._i18n.retrieve("planning", "observation_user_prompt").format(
            task_description=task_desc,
            task_goal=task_goal,
            completed_summary=completed_summary,
            step_number=completed_step.step_number,
            step_description=completed_step.description,
            step_result=result,
            remaining_summary=remaining_summary,
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
