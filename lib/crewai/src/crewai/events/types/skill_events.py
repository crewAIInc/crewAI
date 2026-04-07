"""Skill lifecycle events for the Agent Skills standard.

Events emitted during skill discovery, loading, and activation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from crewai.events.base_events import BaseEvent


class SkillEvent(BaseEvent):
    """Base event for skill operations."""

    skill_name: str = ""
    skill_path: Path | None = None
    from_agent: Any | None = None
    from_task: Any | None = None

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._set_agent_params(data)
        self._set_task_params(data)


class SkillDiscoveryStartedEvent(SkillEvent):
    """Event emitted when skill discovery begins."""

    type: Literal["skill_discovery_started"] = "skill_discovery_started"
    search_path: Path


class SkillDiscoveryCompletedEvent(SkillEvent):
    """Event emitted when skill discovery completes."""

    type: Literal["skill_discovery_completed"] = "skill_discovery_completed"
    search_path: Path
    skills_found: int
    skill_names: list[str]


class SkillLoadedEvent(SkillEvent):
    """Event emitted when a skill is loaded at metadata level."""

    type: Literal["skill_loaded"] = "skill_loaded"
    disclosure_level: int = 1


class SkillActivatedEvent(SkillEvent):
    """Event emitted when a skill is activated (promoted to instructions level)."""

    type: Literal["skill_activated"] = "skill_activated"
    disclosure_level: int = 2


class SkillLoadFailedEvent(SkillEvent):
    """Event emitted when skill loading fails."""

    type: Literal["skill_load_failed"] = "skill_load_failed"
    error: str
