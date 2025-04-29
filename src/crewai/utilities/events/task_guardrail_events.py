from typing import Any, Callable, Optional, Union

from crewai.utilities.events.base_events import BaseEvent


class TaskGuardrailStartedEvent(BaseEvent):
    """Event emitted when a guardrail task starts

    Attributes:
        guardrail: The guardrail callable or TaskGuardrail instance
        retry_count: The number of times the guardrail has been retried
    """

    type: str = "task_guardrail_started"
    guardrail: Union[str, Callable]
    retry_count: int

    def __init__(self, **data):
        from inspect import getsource

        from crewai.tasks.task_guardrail import TaskGuardrail

        super().__init__(**data)

        if isinstance(self.guardrail, TaskGuardrail):
            self.guardrail = self.guardrail.description.strip()
        elif isinstance(self.guardrail, Callable):
            self.guardrail = getsource(self.guardrail).strip()


class TaskGuardrailCompletedEvent(BaseEvent):
    """Event emitted when a guardrail task completes"""

    type: str = "task_guardrail_completed"
    success: bool
    result: Any
    error: Optional[str] = None
    retry_count: int
