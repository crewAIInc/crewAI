"""Observation events for the Plan-and-Execute architecture.

Emitted during the Observation phase (PLAN-AND-ACT Section 3.3) when the
PlannerObserver analyzes step execution results and decides on plan
continuation, refinement, or replanning.
"""

from typing import Any

from crewai.events.base_events import BaseEvent


class ObservationEvent(BaseEvent):
    """Base event for observation phase events."""

    type: str
    agent_role: str
    step_number: int
    step_description: str = ""
    from_task: Any | None = None
    from_agent: Any | None = None

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._set_task_params(data)
        self._set_agent_params(data)


class StepObservationStartedEvent(ObservationEvent):
    """Emitted when the Planner begins observing a step's result.

    Fires after every step execution, before the observation LLM call.
    """

    type: str = "step_observation_started"


class StepObservationCompletedEvent(ObservationEvent):
    """Emitted when the Planner finishes observing a step's result.

    Contains the full observation analysis: what was learned, whether
    the plan is still valid, and what action to take next.
    """

    type: str = "step_observation_completed"
    step_completed_successfully: bool = True
    key_information_learned: str = ""
    remaining_plan_still_valid: bool = True
    needs_full_replan: bool = False
    replan_reason: str | None = None
    goal_already_achieved: bool = False
    suggested_refinements: list[str] | None = None


class StepObservationFailedEvent(ObservationEvent):
    """Emitted when the observation LLM call itself fails.

    The system defaults to continuing the plan when this happens,
    but the event allows monitoring/alerting on observation failures.
    """

    type: str = "step_observation_failed"
    error: str = ""


class PlanRefinementEvent(ObservationEvent):
    """Emitted when the Planner refines upcoming step descriptions.

    This is the lightweight refinement path — no full replan, just
    sharpening pending todo descriptions based on new information.
    """

    type: str = "plan_refinement"
    refined_step_count: int = 0
    refinements: list[str] | None = None


class PlanReplanTriggeredEvent(ObservationEvent):
    """Emitted when the Planner triggers a full replan.

    The remaining plan was deemed fundamentally wrong and will be
    regenerated from scratch, preserving completed step results.
    """

    type: str = "plan_replan_triggered"
    replan_reason: str = ""
    replan_count: int = 0
    completed_steps_preserved: int = 0


class GoalAchievedEarlyEvent(ObservationEvent):
    """Emitted when the Planner detects the goal was achieved early.

    Remaining steps will be skipped and execution will finalize.
    """

    type: str = "goal_achieved_early"
    steps_remaining: int = 0
    steps_completed: int = 0
