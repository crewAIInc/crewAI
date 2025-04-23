from typing import Any, Callable, Optional, Union

from crewai.utilities.events.base_events import BaseEvent


class TaskGuardrailStartedEvent(BaseEvent):
    """Event emitted when a guardrail task starts

    Attributes:
        messages: Content can be either a string or a list of dictionaries that support
            multimodal content (text, images, etc.)
    """

    type: str = "task_guardrail_started"
    guardrail: Union[str, Callable]
    retry_count: int

    def __init__(self, **data):
        from inspect import getsource

        from crewai.tasks.task_guardrail import TaskGuardrail

        super().__init__(**data)

        if isinstance(self.guardrail, TaskGuardrail):
            assert self.guardrail.generated_code is not None
            self.guardrail = self.guardrail.generated_code.strip()
        elif isinstance(self.guardrail, Callable):
            self.guardrail = getsource(self.guardrail).strip()


class TaskGuardrailCompletedEvent(BaseEvent):
    """Event emitted when a guardrail task completes"""

    type: str = "task_guardrail_completed"
    success: bool
    result: Any
    error: Optional[str] = None
    retry_count: int
