from typing import Any

from crewai.events.base_events import BaseEvent


class MemoryBaseEvent(BaseEvent):
    """Base event for memory operations"""

    type: str
    task_id: str | None = None
    task_name: str | None = None
    from_task: Any | None = None
    from_agent: Any | None = None
    agent_role: str | None = None
    agent_id: str | None = None

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._set_agent_params(data)
        self._set_task_params(data)


class MemoryQueryStartedEvent(MemoryBaseEvent):
    """Event emitted when a memory query is started"""

    type: str = "memory_query_started"
    query: str
    limit: int
    score_threshold: float | None = None


class MemoryQueryCompletedEvent(MemoryBaseEvent):
    """Event emitted when a memory query is completed successfully"""

    type: str = "memory_query_completed"
    query: str
    results: Any
    limit: int
    score_threshold: float | None = None
    query_time_ms: float


class MemoryQueryFailedEvent(MemoryBaseEvent):
    """Event emitted when a memory query fails"""

    type: str = "memory_query_failed"
    query: str
    limit: int
    score_threshold: float | None = None
    error: str


class MemorySaveStartedEvent(MemoryBaseEvent):
    """Event emitted when a memory save operation is started"""

    type: str = "memory_save_started"
    value: str | None = None
    metadata: dict[str, Any] | None = None
    agent_role: str | None = None


class MemorySaveCompletedEvent(MemoryBaseEvent):
    """Event emitted when a memory save operation is completed successfully"""

    type: str = "memory_save_completed"
    value: str
    metadata: dict[str, Any] | None = None
    agent_role: str | None = None
    save_time_ms: float


class MemorySaveFailedEvent(MemoryBaseEvent):
    """Event emitted when a memory save operation fails"""

    type: str = "memory_save_failed"
    value: str | None = None
    metadata: dict[str, Any] | None = None
    agent_role: str | None = None
    error: str


class MemoryRetrievalStartedEvent(MemoryBaseEvent):
    """Event emitted when memory retrieval for a task prompt starts"""

    type: str = "memory_retrieval_started"
    task_id: str | None = None


class MemoryRetrievalCompletedEvent(MemoryBaseEvent):
    """Event emitted when memory retrieval for a task prompt completes successfully"""

    type: str = "memory_retrieval_completed"
    task_id: str | None = None
    memory_content: str
    retrieval_time_ms: float


class MemoryRetrievalFailedEvent(MemoryBaseEvent):
    """Event emitted when memory retrieval for a task prompt fails."""

    type: str = "memory_retrieval_failed"
    task_id: str | None = None
    error: str
