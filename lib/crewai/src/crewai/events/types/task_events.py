from typing import Any, Literal

from crewai.events.base_events import BaseEvent
from crewai.tasks.task_output import TaskOutput


def _set_task_fingerprint(event: BaseEvent, task: Any) -> None:
    """Set task identity and fingerprint data on an event."""
    if task is None:
        return
    task_id = getattr(task, "id", None)
    if task_id is not None:
        event.task_id = str(task_id)
    task_name = getattr(task, "name", None) or getattr(task, "description", None)
    if task_name:
        event.task_name = task_name
    if task.fingerprint:
        event.source_fingerprint = task.fingerprint.uuid_str
        event.source_type = "task"
        if task.fingerprint.metadata:
            event.fingerprint_metadata = task.fingerprint.metadata


class TaskStartedEvent(BaseEvent):
    """Event emitted when a task starts"""

    type: Literal["task_started"] = "task_started"
    context: str | None
    task: Any | None = None

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        _set_task_fingerprint(self, self.task)


class TaskCompletedEvent(BaseEvent):
    """Event emitted when a task completes"""

    output: TaskOutput
    type: Literal["task_completed"] = "task_completed"
    task: Any | None = None

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        _set_task_fingerprint(self, self.task)


class TaskFailedEvent(BaseEvent):
    """Event emitted when a task fails"""

    error: str
    type: Literal["task_failed"] = "task_failed"
    task: Any | None = None

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        _set_task_fingerprint(self, self.task)


class TaskEvaluationEvent(BaseEvent):
    """Event emitted when a task evaluation is completed"""

    type: Literal["task_evaluation"] = "task_evaluation"
    evaluation_type: str
    task: Any | None = None

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        _set_task_fingerprint(self, self.task)
