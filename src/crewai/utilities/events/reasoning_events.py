from crewai.utilities.events.base_events import BaseEvent


class AgentReasoningStartedEvent(BaseEvent):
    """Event emitted when an agent starts reasoning about a task."""

    type: str = "agent_reasoning_started"
    agent_role: str
    task_id: str
    attempt: int = 1  # The current reasoning/refinement attempt


class AgentReasoningCompletedEvent(BaseEvent):
    """Event emitted when an agent finishes its reasoning process."""

    type: str = "agent_reasoning_completed"
    agent_role: str
    task_id: str
    plan: str
    ready: bool
    attempt: int = 1


class AgentReasoningFailedEvent(BaseEvent):
    """Event emitted when the reasoning process fails."""

    type: str = "agent_reasoning_failed"
    agent_role: str
    task_id: str
    error: str
    attempt: int = 1