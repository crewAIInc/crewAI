from typing import Any, Literal

from crewai.events.base_events import BaseEvent


class KnowledgeEventBase(BaseEvent):
    task_id: str | None = None
    task_name: str | None = None
    from_task: Any | None = None
    from_agent: Any | None = None
    agent_role: str | None = None
    agent_id: str | None = None

    def __init__(self, **data):
        super().__init__(**data)
        self._set_agent_params(data)
        self._set_task_params(data)


class KnowledgeRetrievalStartedEvent(KnowledgeEventBase):
    """Event emitted when a knowledge retrieval is started."""

    type: Literal["knowledge_search_query_started"] = "knowledge_search_query_started"


class KnowledgeRetrievalCompletedEvent(KnowledgeEventBase):
    """Event emitted when a knowledge retrieval is completed."""

    query: str
    type: Literal["knowledge_search_query_completed"] = "knowledge_search_query_completed"
    retrieved_knowledge: str


class KnowledgeQueryStartedEvent(KnowledgeEventBase):
    """Event emitted when a knowledge query is started."""

    task_prompt: str
    type: Literal["knowledge_query_started"] = "knowledge_query_started"


class KnowledgeQueryFailedEvent(KnowledgeEventBase):
    """Event emitted when a knowledge query fails."""

    type: Literal["knowledge_query_failed"] = "knowledge_query_failed"
    error: str


class KnowledgeQueryCompletedEvent(KnowledgeEventBase):
    """Event emitted when a knowledge query is completed."""

    query: str
    type: Literal["knowledge_query_completed"] = "knowledge_query_completed"


class KnowledgeSearchQueryFailedEvent(KnowledgeEventBase):
    """Event emitted when a knowledge search query fails."""

    query: str
    type: Literal["knowledge_search_query_failed"] = "knowledge_search_query_failed"
    error: str
