"""Agent logging events that don't reference BaseAgent to avoid circular imports."""

from typing import Any

from pydantic import ConfigDict

from crewai.events.base_events import BaseEvent


class AgentLogsStartedEvent(BaseEvent):
    """Event emitted when agent logs should be shown at start"""

    agent_role: str
    task_description: str | None = None
    verbose: bool = False
    type: str = "agent_logs_started"


class AgentLogsExecutionEvent(BaseEvent):
    """Event emitted when agent logs should be shown during execution"""

    agent_role: str
    formatted_answer: Any
    verbose: bool = False
    type: str = "agent_logs_execution"

    model_config = ConfigDict(arbitrary_types_allowed=True)
