from typing import Any

from crewai.events.base_events import BaseEvent
from crewai.tasks.task_output import TaskOutput


class TaskStartedEvent(BaseEvent):
    """Event emitted when a task starts"""

    type: str = "task_started"
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
    type: str = "task_completed"
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
    type: str = "task_failed"
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

    type: str = "task_evaluation"
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
