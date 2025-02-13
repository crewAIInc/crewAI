from typing import Any, Optional

from crewai.utilities.events.crew_events import CrewEvent


class TaskStartedEvent(CrewEvent):
    """Event emitted when a task starts"""

    type: str = "task_started"
    context: Optional[str]
    model_config = {"arbitrary_types_allowed": True}



class TaskCompletedEvent(CrewEvent):
    """Event emitted when a task completes"""

    output: Any
    type: str = "task_completed"

    model_config = {"arbitrary_types_allowed": True}


class TaskFailedEvent(CrewEvent):
    """Event emitted when a task fails"""

    task: Any
    error: str
    type: str = "task_failed"
