from typing import Any, Literal

from crewai.events.base_events import BaseEvent
from crewai.tasks.task_output import TaskOutput


class TaskStartedEvent(BaseEvent):
    """Event emitted when a task starts"""

    type: Literal["task_started"] = "task_started"
    context: str | None
    task: Any | None = None

    def __init__(self, **data):
        super().__init__(**data)
        # Set fingerprint data from the task
        if hasattr(self.task, "fingerprint") and self.task.fingerprint:
            self.source_fingerprint = self.task.fingerprint.uuid_str
            self.source_type = "task"
            if (
                hasattr(self.task.fingerprint, "metadata")
                and self.task.fingerprint.metadata
            ):
                self.fingerprint_metadata = self.task.fingerprint.metadata


class TaskCompletedEvent(BaseEvent):
    """Event emitted when a task completes"""

    output: TaskOutput
    type: Literal["task_completed"] = "task_completed"
    task: Any | None = None

    def __init__(self, **data):
        super().__init__(**data)
        # Set fingerprint data from the task
        if hasattr(self.task, "fingerprint") and self.task.fingerprint:
            self.source_fingerprint = self.task.fingerprint.uuid_str
            self.source_type = "task"
            if (
                hasattr(self.task.fingerprint, "metadata")
                and self.task.fingerprint.metadata
            ):
                self.fingerprint_metadata = self.task.fingerprint.metadata


class TaskFailedEvent(BaseEvent):
    """Event emitted when a task fails"""

    error: str
    type: Literal["task_failed"] = "task_failed"
    task: Any | None = None

    def __init__(self, **data):
        super().__init__(**data)
        # Set fingerprint data from the task
        if hasattr(self.task, "fingerprint") and self.task.fingerprint:
            self.source_fingerprint = self.task.fingerprint.uuid_str
            self.source_type = "task"
            if (
                hasattr(self.task.fingerprint, "metadata")
                and self.task.fingerprint.metadata
            ):
                self.fingerprint_metadata = self.task.fingerprint.metadata


class TaskEvaluationEvent(BaseEvent):
    """Event emitted when a task evaluation is completed"""

    type: Literal["task_evaluation"] = "task_evaluation"
    evaluation_type: str
    task: Any | None = None

    def __init__(self, **data):
        super().__init__(**data)
        # Set fingerprint data from the task
        if hasattr(self.task, "fingerprint") and self.task.fingerprint:
            self.source_fingerprint = self.task.fingerprint.uuid_str
            self.source_type = "task"
            if (
                hasattr(self.task.fingerprint, "metadata")
                and self.task.fingerprint.metadata
            ):
                self.fingerprint_metadata = self.task.fingerprint.metadata
