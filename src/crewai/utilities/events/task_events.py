from typing import Optional

from crewai.tasks.task_output import TaskOutput
from crewai.utilities.events.base_events import CrewEvent


class TaskStartedEvent(CrewEvent):
    """Event emitted when a task starts"""

    type: str = "task_started"
    context: Optional[str]


class TaskCompletedEvent(CrewEvent):
    """Event emitted when a task completes"""

    output: TaskOutput
    type: str = "task_completed"


class TaskFailedEvent(CrewEvent):
    """Event emitted when a task fails"""

    error: str
    type: str = "task_failed"


class TaskEvaluationEvent(CrewEvent):
    """Event emitted when a task evaluation is completed"""

    type: str = "task_evaluation"
    evaluation_type: str
