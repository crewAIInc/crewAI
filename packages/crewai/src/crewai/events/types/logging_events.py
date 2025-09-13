"""Agent logging events that don't reference BaseAgent to avoid circular imports."""

from typing import Any, Optional

from crewai.events.base_events import BaseEvent


class AgentLogsStartedEvent(BaseEvent):
    """Event emitted when agent logs should be shown at start"""

    agent_role: str
    task_description: Optional[str] = None
    verbose: bool = False
    type: str = "agent_logs_started"


class AgentLogsExecutionEvent(BaseEvent):
    """Event emitted when agent logs should be shown during execution"""

    agent_role: str
    formatted_answer: Any
    verbose: bool = False
    type: str = "agent_logs_execution"

    model_config = {"arbitrary_types_allowed": True}
