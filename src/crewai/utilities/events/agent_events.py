from typing import Any

from .crew_events import CrewEvent


class AgentExecutionStarted(CrewEvent):
    """Event emitted when an agent starts executing a task"""

    agent: Any  # type: ignore
    task: Any  # type: ignore
    type: str = "agent_execution_started"

    model_config = {"arbitrary_types_allowed": True}


class AgentExecutionCompleted(CrewEvent):
    """Event emitted when an agent completes executing a task"""

    agent: Any
    task: Any
    output: str
    type: str = "agent_execution_completed"

    model_config = {"arbitrary_types_allowed": True}


class AgentExecutionError(CrewEvent):
    """Event emitted when an agent encounters an error during execution"""

    agent: Any
    task: Any
    error: str
    type: str = "agent_execution_error"

    model_config = {"arbitrary_types_allowed": True}
