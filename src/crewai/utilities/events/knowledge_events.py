from typing import TYPE_CHECKING, Any

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.utilities.events.base_events import BaseEvent

if TYPE_CHECKING:
    from crewai.agents.agent_builder.base_agent import BaseAgent


class KnowledgeRetrievalStartedEvent(BaseEvent):
    """Event emitted when a knowledge retrieval is started."""

    type: str = "knowledge_search_query_started"
    agent: BaseAgent


class KnowledgeRetrievalCompletedEvent(BaseEvent):
    """Event emitted when a knowledge retrieval is completed."""

    query: str
    type: str = "knowledge_search_query_completed"
    agent: BaseAgent
    retrieved_knowledge: Any


class KnowledgeQueryStartedEvent(BaseEvent):
    """Event emitted when a knowledge query is started."""

    task_prompt: str
    type: str = "knowledge_query_started"
    agent: BaseAgent


class KnowledgeQueryFailedEvent(BaseEvent):
    """Event emitted when a knowledge query fails."""

    type: str = "knowledge_query_failed"
    agent: BaseAgent
    error: str


class KnowledgeQueryCompletedEvent(BaseEvent):
    """Event emitted when a knowledge query is completed."""

    query: str
    type: str = "knowledge_query_completed"
    agent: BaseAgent


class KnowledgeSearchQueryFailedEvent(BaseEvent):
    """Event emitted when a knowledge search query fails."""

    query: str
    type: str = "knowledge_search_query_failed"
    agent: BaseAgent
    error: str
