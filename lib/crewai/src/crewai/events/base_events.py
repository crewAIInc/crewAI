from collections.abc import Iterator
import contextvars
from datetime import datetime, timezone
import itertools
from typing import Any
import uuid

from pydantic import BaseModel, Field

from crewai.utilities.serialization import Serializable, to_serializable


_emission_counter: contextvars.ContextVar[Iterator[int]] = contextvars.ContextVar(
    "_emission_counter"
)


def _get_or_create_counter() -> Iterator[int]:
    """Get the emission counter for the current context, creating if needed."""
    try:
        return _emission_counter.get()
    except LookupError:
        counter: Iterator[int] = itertools.count(start=1)
        _emission_counter.set(counter)
        return counter


def get_next_emission_sequence() -> int:
    """Get the next emission sequence number.

    Returns:
        The next sequence number.
    """
    return next(_get_or_create_counter())


def reset_emission_counter() -> None:
    """Reset the emission sequence counter to 1.

    Resets for the current context only.
    """
    counter: Iterator[int] = itertools.count(start=1)
    _emission_counter.set(counter)


class BaseEvent(BaseModel):
    """Base class for all events"""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    type: str
    source_fingerprint: str | None = None  # UUID string of the source entity
    source_type: str | None = (
        None  # "agent", "task", "crew", "memory", "entity_memory", "short_term_memory", "long_term_memory", "external_memory"
    )
    fingerprint_metadata: dict[str, Any] | None = None  # Any relevant metadata

    task_id: str | None = None
    task_name: str | None = None
    agent_id: str | None = None
    agent_role: str | None = None

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_event_id: str | None = None
    previous_event_id: str | None = None
    triggered_by_event_id: str | None = None
    emission_sequence: int | None = None

    def to_json(self, exclude: set[str] | None = None) -> Serializable:
        """
        Converts the event to a JSON-serializable dictionary.

        Args:
            exclude (set[str], optional): Set of keys to exclude from the result. Defaults to None.

        Returns:
            dict: A JSON-serializable dictionary.
        """
        return to_serializable(self, exclude=exclude)

    def _set_task_params(self, data: dict[str, Any]) -> None:
        if "from_task" in data and (task := data["from_task"]):
            self.task_id = str(task.id)
            self.task_name = task.name or task.description
            self.from_task = None

    def _set_agent_params(self, data: dict[str, Any]) -> None:
        task = data.get("from_task", None)
        agent = task.agent if task else data.get("from_agent", None)

        if not agent:
            return

        self.agent_id = str(agent.id)
        self.agent_role = agent.role
        self.from_agent = None
