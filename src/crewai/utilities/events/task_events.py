from typing import Any, Optional

from crewai.tasks.task_output import TaskOutput
from crewai.utilities.events.base_events import BaseEvent


class TaskStartedEvent(BaseEvent):
    """Event emitted when a task starts"""

    type: str = "task_started"
    context: Optional[str]
    task: Optional[Any] = None

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
    task: Optional[Any] = None

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
    task: Optional[Any] = None

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
    task: Optional[Any] = None

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
