"""Filesystem discovery and progressive loading for Agent Skills.

Provides functions to discover skills in directories, activate them
for agent use, and format skill context for prompt injection.
"""

from __future__ import annotations

from collections.abc import Iterable
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.skill_events import (
    SkillActivatedEvent,
    SkillDiscoveryCompletedEvent,
    SkillDiscoveryStartedEvent,
    SkillLoadFailedEvent,
    SkillLoadedEvent,
)
from crewai.skills.models import INSTRUCTIONS, RESOURCES, Skill, SkillFrontmatter
from crewai.skills.parser import (
    SKILL_FILENAME,
    load_skill_instructions,
    load_skill_metadata,
    load_skill_resources,
    parse_frontmatter,
)


if TYPE_CHECKING:
    from crewai.agents.agent_builder.base_agent import BaseAgent

_logger = logging.getLogger(__name__)


def discover_skills(
    search_path: Path,
    source: BaseAgent | None = None,
) -> list[Skill]:
    """Scan a directory for skill directories containing SKILL.md.

    Loads each discovered skill at METADATA disclosure level.

    Args:
        search_path: Directory to scan for skill subdirectories.
        source: Optional event source (agent or crew) for event emission.

    Returns:
        List of Skill instances at METADATA level.
    """
    if not search_path.is_dir():
        msg = f"Skill search path does not exist or is not a directory: {search_path}"
        raise FileNotFoundError(msg)

    skills: list[Skill] = []

    if source is not None:
        crewai_event_bus.emit(
            source,
            event=SkillDiscoveryStartedEvent(
                from_agent=source,
                search_path=search_path,
            ),
        )

    for child in sorted(search_path.iterdir()):
        if not child.is_dir():
            continue
        skill_md = child / SKILL_FILENAME
        if not skill_md.is_file():
            continue
        try:
            skill = load_skill_metadata(child)
            skills.append(skill)
            if source is not None:
                crewai_event_bus.emit(
                    source,
                    event=SkillLoadedEvent(
                        from_agent=source,
                        skill_name=skill.name,
                        skill_path=skill.path,
                        disclosure_level=skill.disclosure_level,
                    ),
                )
        except Exception as e:
            _logger.warning("Failed to load skill from %s: %s", child, e)
            if source is not None:
                crewai_event_bus.emit(
                    source,
                    event=SkillLoadFailedEvent(
                        from_agent=source,
                        skill_name=child.name,
                        skill_path=child,
                        error=str(e),
                    ),
                )

    if source is not None:
        crewai_event_bus.emit(
            source,
            event=SkillDiscoveryCompletedEvent(
                from_agent=source,
                search_path=search_path,
                skills_found=len(skills),
                skill_names=[s.name for s in skills],
            ),
        )

    return skills


def activate_skill(
    skill: Skill,
    source: BaseAgent | None = None,
) -> Skill:
    """Promote a skill to INSTRUCTIONS disclosure level.

    Idempotent: returns the skill unchanged if already at or above INSTRUCTIONS.

    Args:
        skill: Skill to activate.
        source: Optional event source for event emission.

    Returns:
        Skill at INSTRUCTIONS level or higher.
    """
    if skill.disclosure_level >= INSTRUCTIONS:
        return skill

    activated = load_skill_instructions(skill)

    if source is not None:
        crewai_event_bus.emit(
            source,
            event=SkillActivatedEvent(
                from_agent=source,
                skill_name=activated.name,
                skill_path=activated.path,
                disclosure_level=activated.disclosure_level,
            ),
        )

    return activated


def load_skill(
    skill: Path | Skill | str,
    source: BaseAgent | None = None,
) -> list[Skill]:
    """Load one skill input into Skill objects.

    Accepts a pre-loaded Skill object, skill search path, inline SKILL.md
    string, or '@org/name' registry reference. Path inputs can expand to many
    skills. Path and inline inputs are activated immediately; pre-loaded Skill
    objects keep their disclosure level.
    """
    if isinstance(skill, Skill):
        return [skill]
    if isinstance(skill, Path):
        return [
            activate_skill(s, source=source)
            for s in discover_skills(skill, source=source)
        ]
    if isinstance(skill, str) and skill.startswith("@"):
        from crewai.skills.registry import resolve_registry_ref

        return [resolve_registry_ref(skill, source=source)]
    if isinstance(skill, str) and skill.lstrip().startswith("---\n"):
        frontmatter_dict, body = parse_frontmatter(skill.strip())
        return [
            Skill(
                frontmatter=SkillFrontmatter(**frontmatter_dict),
                instructions=body,
                path=Path("."),
                disclosure_level=INSTRUCTIONS,
            )
        ]
    if isinstance(skill, str):
        return [
            activate_skill(s, source=source)
            for s in discover_skills(Path(skill), source=source)
        ]

    msg = f"Unsupported skill input: {skill!r}"
    raise TypeError(msg)


def load_skills(
    skills: Iterable[Path | Skill | str],
    source: BaseAgent | None = None,
) -> list[Skill]:
    """Load skill inputs into de-duplicated Skill objects.

    Preserves first-seen order when multiple inputs resolve to the same skill
    name. Registry refs are scoped by org so different orgs can publish skills
    that share a frontmatter name.
    """
    loaded: dict[str, Skill] = {}
    for skill_input in skills:
        for skill in load_skill(skill_input, source=source):
            dedup_key = skill.name
            if isinstance(skill_input, str) and skill_input.startswith("@"):
                from crewai.skills.registry import parse_registry_ref

                org, _ = parse_registry_ref(skill_input)
                dedup_key = f"{org}/{skill.name}"
            if dedup_key not in loaded:
                loaded[dedup_key] = skill
    return list(loaded.values())


def load_resources(skill: Skill) -> Skill:
    """Promote a skill to RESOURCES disclosure level.

    Args:
        skill: Skill to promote.

    Returns:
        Skill at RESOURCES level.
    """
    return load_skill_resources(skill)


def format_skill_context(skill: Skill) -> str:
    """Format skill information for agent prompt injection.

    At METADATA level: returns name and description only.
    At INSTRUCTIONS level or above: returns full SKILL.md body.

    Output is wrapped in <skill name="..."> XML tags so the block can serve
    as a stable cache anchor when injected into the system prompt.

    Args:
        skill: The skill to format.

    Returns:
        Formatted skill context string.
    """
    if skill.disclosure_level >= INSTRUCTIONS and skill.instructions:
        parts = [
            f'<skill name="{skill.name}">',
            skill.description,
            "",
            skill.instructions,
        ]
        if skill.disclosure_level >= RESOURCES and skill.resource_files:
            parts.append("")
            parts.append("### Available Resources")
            for dir_name, files in sorted(skill.resource_files.items()):
                if files:
                    parts.append(f"- **{dir_name}/**: {', '.join(files)}")
        parts.append("</skill>")
        return "\n".join(parts)
    return f'<skill name="{skill.name}">\n{skill.description}\n</skill>'
