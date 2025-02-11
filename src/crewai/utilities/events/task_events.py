from typing import Any

from crewai.utilities.events.crew_events import CrewEvent


class TaskStarted(CrewEvent):
    """Event emitted when a task starts"""

    task: Any
    type: str = "task_started"

    model_config = {"arbitrary_types_allowed": True}


class TaskCompleted(CrewEvent):
    """Event emitted when a task completes"""

    task: Any
    output: Any
    type: str = "task_completed"

    model_config = {"arbitrary_types_allowed": True}


class TaskFailed(CrewEvent):
    """Event emitted when a task fails"""

    task: Any
    error: str
    type: str = "task_failed"
