"""Artifact events."""

from __future__ import annotations

from typing import Any, Literal

from crewai.events.base_events import BaseEvent


class ArtifactCreatedEvent(BaseEvent):
    """Emitted when a run saves a new artifact version."""

    type: Literal["artifact_created"] = "artifact_created"
    id: str
    identifier: str
    version: int
    title: str
    kind: str
    content_type: str
    size: int
    url: str
    from_agent: Any | None = None
    from_task: Any | None = None

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._set_agent_params(data)
        self._set_task_params(data)
