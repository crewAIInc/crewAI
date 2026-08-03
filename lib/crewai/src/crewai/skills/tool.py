"""Runtime tool for progressively disclosing Agent Skill instructions."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any, Final

from pydantic import BaseModel, Field

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.skill_events import SkillUsedEvent
from crewai.skills.loader import (
    activate_skill,
    build_skill_catalog,
    format_skill_context,
)
from crewai.skills.models import Skill
from crewai.tools.base_tool import BaseTool
from crewai.utilities.string_utils import sanitize_tool_name


LOAD_SKILL_TOOL_NAME: Final[str] = "load_skill"


class LoadSkillSchema(BaseModel):
    """Arguments accepted by the runtime skill loader."""

    skill_name: str = Field(
        ...,
        description="Exact name of the available skill whose instructions are needed.",
    )


class LoadSkillTool(BaseTool):
    """Load one relevant skill's instructions for the current execution."""

    name: str = LOAD_SKILL_TOOL_NAME
    description: str = (
        "Load one available skill's full instructions. Most requests need no "
        "skill, so only call this when an available skill's description "
        "clearly matches the current request."
    )
    args_schema: type[BaseModel] = LoadSkillSchema
    catalog: dict[str, Skill] = Field(default_factory=dict, exclude=True)
    source: Any = Field(default=None, exclude=True)
    task: Any = Field(default=None, exclude=True)

    def _run(self, skill_name: str, **kwargs: Any) -> str:
        """Load and return one skill's instruction block."""
        skill = self.catalog.get(skill_name)
        if skill is None:
            available = ", ".join(self.catalog)
            return (
                f"Skill {skill_name!r} is not available. "
                f"Available skills: {available or 'none'}."
            )

        activated = activate_skill(skill, source=self.source)
        crewai_event_bus.emit(
            self.source,
            event=SkillUsedEvent(
                from_agent=self.source,
                from_task=self.task,
                skill_name=skill_name,
                skill_path=activated.path,
                disclosure_level=activated.disclosure_level,
            ),
        )
        return format_skill_context(activated, label=skill_name)


def resolve_loader_tool_name(reserved_names: Collection[str] = ()) -> str:
    """Return a loader tool name that none of *reserved_names* already claims.

    A configured tool holding the default name would otherwise be
    indistinguishable from the loader, so the model could never reach the
    loader the skill catalog points it at.
    """
    taken = {sanitize_tool_name(name) for name in reserved_names}
    if LOAD_SKILL_TOOL_NAME not in taken:
        return LOAD_SKILL_TOOL_NAME

    attempt = 2
    while f"{LOAD_SKILL_TOOL_NAME}_{attempt}" in taken:
        attempt += 1
    return f"{LOAD_SKILL_TOOL_NAME}_{attempt}"


def create_skill_loader_tool(
    skills: list[Skill] | None,
    *,
    source: Any = None,
    task: Any = None,
    reserved_names: Collection[str] = (),
) -> LoadSkillTool | None:
    """Create a loader tool when metadata-only skills are available.

    Args:
        skills: Skills configured on the agent.
        source: Event source for the skill events the loader emits.
        task: Task the loader is scoped to.
        reserved_names: Names of tools already configured on the agent, which
            the loader must not collide with.

    Returns:
        The loader tool, or None when every skill is already disclosed.
    """
    catalog = build_skill_catalog(skills or [])
    if not catalog:
        return None
    return LoadSkillTool(
        name=resolve_loader_tool_name(reserved_names),
        catalog=catalog,
        source=source,
        task=task,
    )
