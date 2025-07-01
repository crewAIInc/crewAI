from typing import Any, Callable, Optional, Tuple, Union

from pydantic import BaseModel, field_validator

class GuardrailResult(BaseModel):
    """Result from a task guardrail execution.

    This class standardizes the return format of task guardrails,
    converting tuple responses into a structured format that can
    be easily handled by the task execution system.

    Attributes:
        success (bool): Whether the guardrail validation passed
        result (Any, optional): The validated/transformed result if successful
        error (str, optional): Error message if validation failed
    """
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None

    @field_validator("result", "error")
    @classmethod
    def validate_result_error_exclusivity(cls, v: Any, info) -> Any:
        values = info.data
        if "success" in values:
            if values["success"] and v and "error" in values and values["error"]:
                raise ValueError("Cannot have both result and error when success is True")
            if not values["success"] and v and "result" in values and values["result"]:
                raise ValueError("Cannot have both result and error when success is False")
        return v

    @classmethod
    def from_tuple(cls, result: Tuple[bool, Union[Any, str]]) -> "GuardrailResult":
        """Create a GuardrailResult from a validation tuple.

        Args:
            result: A tuple of (success, data) where data is either
                   the validated result or error message.

        Returns:
            GuardrailResult: A new instance with the tuple data.
        """
        success, data = result
        return cls(
            success=success,
            result=data if success else None,
            error=data if not success else None
        )


def process_guardrail(output: Any, guardrail: Callable, retry_count: int) -> GuardrailResult:
    """Process the guardrail for the agent output.

    Args:
        output: The output to validate with the guardrail

    Returns:
        GuardrailResult: The result of the guardrail validation
    """
    from crewai.task import TaskOutput
    from crewai.lite_agent import LiteAgentOutput

    assert isinstance(output, TaskOutput) or isinstance(output, LiteAgentOutput), "Output must be a TaskOutput or LiteAgentOutput"

    assert guardrail is not None

    from crewai.utilities.events import (
        LLMGuardrailCompletedEvent,
        LLMGuardrailStartedEvent,
    )
    from crewai.utilities.events.crewai_event_bus import crewai_event_bus

    crewai_event_bus.emit(
        None,
        LLMGuardrailStartedEvent(
            guardrail=guardrail, retry_count=retry_count
        ),
    )

    result = guardrail(output)
    guardrail_result = GuardrailResult.from_tuple(result)

    crewai_event_bus.emit(
        None,
        LLMGuardrailCompletedEvent(
            success=guardrail_result.success,
            result=guardrail_result.result,
            error=guardrail_result.error,
            retry_count=retry_count,
        ),
    )

    return guardrail_result
