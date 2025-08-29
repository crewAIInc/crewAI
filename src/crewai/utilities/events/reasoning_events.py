from crewai.utilities.events.base_events import BaseEvent
from typing import Any, Optional


class ReasoningEvent(BaseEvent):
    """Base event for reasoning events."""

    type: str
    attempt: int = 1
    agent_role: str
    task_id: str
    task_name: Optional[str] = None
    from_task: Optional[Any] = None
    agent_id: Optional[str] = None
    from_agent: Optional[Any] = None

    def __init__(self, **data):
        super().__init__(**data)
        self._set_task_params(data)
        self._set_agent_params(data)


class AgentReasoningStartedEvent(ReasoningEvent):
    """Event emitted when an agent starts reasoning about a task."""

    type: str = "agent_reasoning_started"
    agent_role: str
    task_id: str


class AgentReasoningCompletedEvent(ReasoningEvent):
    """Event emitted when an agent finishes its reasoning process."""

    type: str = "agent_reasoning_completed"
    agent_role: str
    task_id: str
    plan: str
    ready: bool


class AgentReasoningFailedEvent(ReasoningEvent):
    """Event emitted when the reasoning process fails."""

    type: str = "agent_reasoning_failed"
    agent_role: str
    task_id: str
    error: str
