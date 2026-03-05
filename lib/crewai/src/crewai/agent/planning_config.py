from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class PlanningConfig(BaseModel):
    """Configuration for agent planning/reasoning before task execution.

    This allows users to customize the planning behavior including prompts,
    iteration limits, the LLM used for planning, and the reasoning effort
    level that controls post-step observation and replanning behavior.

    Note: To disable planning, don't pass a planning_config or set planning=False
    on the Agent. The presence of a PlanningConfig enables planning.

    Attributes:
        reasoning_effort: Controls observation and replanning after each step.
            - "low": Observe each step (validates success), but skip the
              decide/replan/refine pipeline. Steps are marked complete and
              execution continues linearly. Fastest option.
            - "medium": Observe each step. On failure, trigger replanning.
              On success, skip refinement and continue. Balanced option.
            - "high": Full observation pipeline — observe every step, then
              route through decide_next_action which can trigger early goal
              achievement, full replanning, or lightweight refinement.
              Most adaptive but adds latency per step.
        max_attempts: Maximum number of planning refinement attempts.
            If None, will continue until the agent indicates readiness.
        max_steps: Maximum number of steps in the generated plan.
        system_prompt: Custom system prompt for planning. Uses default if None.
        plan_prompt: Custom prompt for creating the initial plan.
        refine_prompt: Custom prompt for refining the plan.
        llm: LLM to use for planning. Uses agent's LLM if None.

    Example:
        ```python
        from crewai import Agent
        from crewai.agent.planning_config import PlanningConfig

        # Simple usage — fast, linear execution (default)
        agent = Agent(
            role="Researcher",
            goal="Research topics",
            backstory="Expert researcher",
            planning_config=PlanningConfig(),
        )

        # Balanced — replan only when steps fail
        agent = Agent(
            role="Researcher",
            goal="Research topics",
            backstory="Expert researcher",
            planning_config=PlanningConfig(
                reasoning_effort="medium",
            ),
        )

        # Full adaptive planning with refinement and replanning
        agent = Agent(
            role="Researcher",
            goal="Research topics",
            backstory="Expert researcher",
            planning_config=PlanningConfig(
                reasoning_effort="high",
                max_attempts=3,
                max_steps=10,
                plan_prompt="Create a focused plan for: {description}",
                llm="gpt-4o-mini",  # Use cheaper model for planning
            ),
        )
        ```
    """

    reasoning_effort: Literal["low", "medium", "high"] = Field(
        default="low",
        description=(
            "Controls post-step observation and replanning behavior. "
            "'low' observes steps but skips replanning/refinement (fastest). "
            "'medium' observes and replans only on step failure (balanced). "
            "'high' runs full observation pipeline with replanning, refinement, "
            "and early goal detection (most adaptive, highest latency)."
        ),
    )
    max_attempts: int | None = Field(
        default=None,
        description=(
            "Maximum number of planning refinement attempts. "
            "If None, will continue until the agent indicates readiness."
        ),
    )
    max_steps: int = Field(
        default=20,
        description="Maximum number of steps in the generated plan.",
        ge=1,
    )
    system_prompt: str | None = Field(
        default=None,
        description="Custom system prompt for planning. Uses default if None.",
    )
    plan_prompt: str | None = Field(
        default=None,
        description="Custom prompt for creating the initial plan.",
    )
    refine_prompt: str | None = Field(
        default=None,
        description="Custom prompt for refining the plan.",
    )
    llm: str | Any | None = Field(
        default=None,
        description="LLM to use for planning. Uses agent's LLM if None.",
    )

    model_config = {"arbitrary_types_allowed": True}
