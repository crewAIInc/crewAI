from typing import Any

from crewai.events.base_events import BaseEvent
from crewai.tasks.task_output import TaskOutput


def _set_task_fingerprint(event: BaseEvent, task: Any) -> None:
    """Set fingerprint data on an event from a task object."""
    if task is not None and task.fingerprint:
        event.source_fingerprint = task.fingerprint.uuid_str
        event.source_type = "task"
        if task.fingerprint.metadata:
            event.fingerprint_metadata = task.fingerprint.metadata


class TaskStartedEvent(BaseEvent):
    """Event emitted when a task starts"""

    type: str = "task_started"
    context: str | None
    task: Any | None = None

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        _set_task_fingerprint(self, self.task)


class TaskCompletedEvent(BaseEvent):
    """Event emitted when a task completes"""

    output: TaskOutput
    type: str = "task_completed"
    task: Any | None = None

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        _set_task_fingerprint(self, self.task)


class TaskFailedEvent(BaseEvent):
    """Event emitted when a task fails"""

    error: str
    type: str = "task_failed"
    task: Any | None = None

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        _set_task_fingerprint(self, self.task)


class TaskEvaluationEvent(BaseEvent):
    """Event emitted when a task evaluation is completed"""

    type: str = "task_evaluation"
    evaluation_type: str
    task: Any | None = None

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        _set_task_fingerprint(self, self.task)
