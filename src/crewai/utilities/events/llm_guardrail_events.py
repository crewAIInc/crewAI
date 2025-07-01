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
        from inspect import getsource

        from crewai.tasks.llm_guardrail import LLMGuardrail
        from crewai.tasks.hallucination_guardrail import HallucinationGuardrail

        super().__init__(**data)

        if isinstance(self.guardrail, LLMGuardrail) or isinstance(
            self.guardrail, HallucinationGuardrail
        ):
            self.guardrail = self.guardrail.description.strip()
        elif isinstance(self.guardrail, Callable):
            self.guardrail = getsource(self.guardrail).strip()


class LLMGuardrailCompletedEvent(BaseEvent):
    """Event emitted when a guardrail task completes"""

    type: str = "llm_guardrail_completed"
    success: bool
    result: Any
    error: Optional[str] = None
    retry_count: int
