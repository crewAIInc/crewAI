from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PlanningConfig(BaseModel):
    """Configuration for agent planning/reasoning before task execution.

    This allows users to customize the planning behavior including prompts,
    iteration limits, and the LLM used for planning.

    Attributes:
        enabled: Whether planning is enabled. Defaults to True.
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

        # Simple usage
        agent = Agent(
            role="Researcher",
            goal="Research topics",
            backstory="Expert researcher",
            planning_config=PlanningConfig(),
        )

        # Customized planning
        agent = Agent(
            role="Researcher",
            goal="Research topics",
            backstory="Expert researcher",
            planning_config=PlanningConfig(
                max_attempts=3,
                max_steps=10,
                plan_prompt="Create a focused plan for: {description}",
                llm="gpt-4o-mini",  # Use cheaper model for planning
            ),
        )
        ```
    """

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
