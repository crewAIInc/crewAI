from collections.abc import Callable
from inspect import getsource
from typing import Any

from crewai.events.base_events import BaseEvent


class LLMGuardrailStartedEvent(BaseEvent):
    """Event emitted when a guardrail task starts

    Attributes:
        guardrail: The guardrail callable or LLMGuardrail instance
        retry_count: The number of times the guardrail has been retried
    """

    type: str = "llm_guardrail_started"
    guardrail: str | Callable
    retry_count: int

    def __init__(self, **data):
        from crewai.tasks.hallucination_guardrail import HallucinationGuardrail
        from crewai.tasks.llm_guardrail import LLMGuardrail

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
    error: str | None = None
    retry_count: int
