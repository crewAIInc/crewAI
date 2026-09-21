"""Download lifecycle events for registry-backed skills.

These events are emitted only by the experimental Skills Repository
(`@org/name` resolution + global cache). Local-file skill events still
live in `crewai.events.types.skill_events`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from crewai.events.types.skill_events import SkillEvent


class SkillDownloadStartedEvent(SkillEvent):
    """Event emitted when a registry skill download begins."""

    type: Literal["skill_download_started"] = "skill_download_started"
    registry_ref: str
    version: str | None = None


class SkillDownloadCompletedEvent(SkillEvent):
    """Event emitted when a registry skill download completes."""

    type: Literal["skill_download_completed"] = "skill_download_completed"
    registry_ref: str
    version: str | None = None
    cache_path: Path | None = None
