"""Runtime tool for progressively disclosing Agent Skill instructions."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.skill_events import SkillUsedEvent
from crewai.skills.loader import activate_skill, format_skill_context
from crewai.skills.models import INSTRUCTIONS, Skill
from crewai.tools.base_tool import BaseTool


class LoadSkillSchema(BaseModel):
    """Arguments accepted by the runtime skill loader."""

    skill_name: str = Field(
        ...,
        description="Exact name of the available skill whose instructions are needed.",
    )


class LoadSkillTool(BaseTool):
    """Load one relevant skill's instructions for the current execution."""

    name: str = "load_skill"
    description: str = (
        "Load the full instructions for one available skill. Use this before "
        "working on a request when an available skill's description applies."
    )
    args_schema: type[BaseModel] = LoadSkillSchema
    skills: list[Skill] = Field(default_factory=list, exclude=True)
    source: Any = Field(default=None, exclude=True)
    task: Any = Field(default=None, exclude=True)

    def _run(self, skill_name: str, **kwargs: Any) -> str:
        """Load and return one skill's instruction block."""
        skill = next((item for item in self.skills if item.name == skill_name), None)
        if skill is None:
            available = ", ".join(item.name for item in self.skills)
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
                skill_name=activated.name,
                skill_path=activated.path,
                disclosure_level=activated.disclosure_level,
            ),
        )
        return format_skill_context(activated)


def create_skill_loader_tool(
    skills: list[Skill] | None,
    *,
    source: Any = None,
    task: Any = None,
) -> LoadSkillTool | None:
    """Create a loader tool when metadata-only skills are available."""
    available = [
        skill for skill in skills or [] if skill.disclosure_level < INSTRUCTIONS
    ]
    if not available:
        return None
    return LoadSkillTool(skills=available, source=source, task=task)
