from typing import Any, Dict, Optional

from crewai.utilities.events.base_events import BaseEvent


class MemoryQueryStartedEvent(BaseEvent):
    """Event emitted when a memory query is started"""

    type: str = "memory_query_started"
    query: str
    limit: int
    score_threshold: Optional[float] = None


class MemoryQueryCompletedEvent(BaseEvent):
    """Event emitted when a memory query is completed successfully"""

    type: str = "memory_query_completed"
    query: str
    results: Any
    limit: int
    score_threshold: Optional[float] = None
    query_time_ms: float


class MemoryQueryFailedEvent(BaseEvent):
    """Event emitted when a memory query fails"""

    type: str = "memory_query_failed"
    query: str
    limit: int
    score_threshold: Optional[float] = None
    error: str


class MemorySaveStartedEvent(BaseEvent):
    """Event emitted when a memory save operation is started"""

    type: str = "memory_save_started"
    value: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    agent_role: Optional[str] = None


class MemorySaveCompletedEvent(BaseEvent):
    """Event emitted when a memory save operation is completed successfully"""

    type: str = "memory_save_completed"
    value: str
    metadata: Optional[Dict[str, Any]] = None
    agent_role: Optional[str] = None
    save_time_ms: float


class MemorySaveFailedEvent(BaseEvent):
    """Event emitted when a memory save operation fails"""

    type: str = "memory_save_failed"
    value: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    agent_role: Optional[str] = None
    error: str


class MemoryRetrievalStartedEvent(BaseEvent):
    """Event emitted when memory retrieval for a task prompt starts"""

    type: str = "memory_retrieval_started"
    task_id: Optional[str] = None


class MemoryRetrievalCompletedEvent(BaseEvent):
    """Event emitted when memory retrieval for a task prompt completes successfully"""

    type: str = "memory_retrieval_completed"
    task_id: Optional[str] = None
    memory_content: str
    retrieval_time_ms: float
