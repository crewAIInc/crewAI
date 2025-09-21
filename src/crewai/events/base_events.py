from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from crewai.utilities.serialization import to_serializable


class BaseEvent(BaseModel):
    """Base class for all events"""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    type: str
    source_fingerprint: str | None = None  # UUID string of the source entity
    source_type: str | None = (
        None  # "agent", "task", "crew", "memory", "entity_memory", "short_term_memory", "long_term_memory", "external_memory"
    )
    fingerprint_metadata: dict[str, Any] | None = None  # Any relevant metadata

    def to_json(self, exclude: set[str] | None = None):
        """
        Converts the event to a JSON-serializable dictionary.

        Args:
            exclude (set[str], optional): Set of keys to exclude from the result. Defaults to None.

        Returns:
            dict: A JSON-serializable dictionary.
        """
        return to_serializable(self, exclude=exclude)

    def _set_task_params(self, data: dict[str, Any]):
        if "from_task" in data and (task := data["from_task"]):
            self.task_id = task.id
            self.task_name = task.name or task.description
            self.from_task = None

    def _set_agent_params(self, data: dict[str, Any]):
        task = data.get("from_task", None)
        agent = task.agent if task else data.get("from_agent", None)

        if not agent:
            return

        self.agent_id = agent.id
        self.agent_role = agent.role
        self.from_agent = None
