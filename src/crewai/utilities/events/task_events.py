from typing import Any

from crewai.utilities.events.crew_events import CrewEvent


class TaskStartedEvent(CrewEvent):
    """Event emitted when a task starts"""

    task: Any
    type: str = "task_started"

    model_config = {"arbitrary_types_allowed": True}


class TaskCompletedEvent(CrewEvent):
    """Event emitted when a task completes"""

    task: Any
    output: Any
    type: str = "task_completed"

    model_config = {"arbitrary_types_allowed": True}


class TaskFailedEvent(CrewEvent):
    """Event emitted when a task fails"""

    task: Any
    error: str
    type: str = "task_failed"
