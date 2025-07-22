from inspect import getsource
from typing import Any, Callable, Optional, Union

from crewai.utilities.events.base_events import BaseEvent


class LLMGuardrailStartedEvent(BaseEvent):
    """Event emitted when a guardrail task starts

    Attributes:
        guardrail: The guardrail callable or LLMGuardrail instance
        retry_count: The number of times the guardrail has been retried
    """

    type: str = "llm_guardrail_started"
    guardrail: Union[str, Callable]
    retry_count: int

    def __init__(self, **data):
        from crewai.tasks.llm_guardrail import LLMGuardrail
        from crewai.tasks.hallucination_guardrail import HallucinationGuardrail

        super().__init__(**data)

        if isinstance(self.guardrail, (LLMGuardrail, HallucinationGuardrail)):
            self.guardrail = self.guardrail.description.strip()
        elif isinstance(self.guardrail, Callable):
            self.guardrail = getsource(self.guardrail).strip()


class LLMGuardrailCompletedEvent(BaseEvent):
    """Event emitted when a guardrail task completes

    Attributes:
        success: Whether the guardrail validation passed
        result: The validation result
        error: Error message if validation failed
        retry_count: The number of times the guardrail has been retried
    """

    type: str = "llm_guardrail_completed"
    success: bool
    result: Any
    error: Optional[str] = None
    retry_count: int
