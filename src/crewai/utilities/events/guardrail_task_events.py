from typing import Any, Callable, Optional, Union

from pydantic import BaseModel

from crewai.utilities.events.base_events import BaseEvent


class GuardrailTaskStartedEvent(BaseEvent):
    """Event emitted when a guardrail task starts

    Attributes:
        messages: Content can be either a string or a list of dictionaries that support
            multimodal content (text, images, etc.)
    """

    type: str = "guardrail_task_started"
    guardrail: Union[str, Callable]
    retry_count: int


class GuardrailTaskCompletedEvent(BaseEvent):
    """Event emitted when a guardrail task completes"""

    type: str = "guardrail_task_completed"
    success: bool
    result: Any
    error: Optional[str] = None
    retry_count: int
