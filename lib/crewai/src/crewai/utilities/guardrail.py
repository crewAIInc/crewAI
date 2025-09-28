from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, field_validator
from typing_extensions import Self

if TYPE_CHECKING:
    from crewai.lite_agent import LiteAgentOutput
    from crewai.tasks.task_output import TaskOutput


class GuardrailResult(BaseModel):
    """Result from a task guardrail execution.

    This class standardizes the return format of task guardrails,
    converting tuple responses into a structured format that can
    be easily handled by the task execution system.

    Attributes:
        success: Whether the guardrail validation passed
        result: The validated/transformed result if successful
        error: Error message if validation failed
    """

    success: bool = Field(description="Whether the guardrail validation passed")
    result: Any | None = Field(
        default=None, description="The validated/transformed result if successful"
    )
    error: str | None = Field(
        default=None, description="Error message if validation failed"
    )

    @field_validator("result", "error")
    @classmethod
    def validate_result_error_exclusivity(cls, v: Any, info) -> Any:
        """Ensure that result and error are mutually exclusive based on success.

        Args:
          v: The value being validated (either result or error)
          info: Validation info containing the entire model data

        Returns:
          The original value if validation passes
        """
        values = info.data
        if "success" in values:
            if values["success"] and v and "error" in values and values["error"]:
                raise ValueError(
                    "Cannot have both result and error when success is True"
                )
            if not values["success"] and v and "result" in values and values["result"]:
                raise ValueError(
                    "Cannot have both result and error when success is False"
                )
        return v

    @classmethod
    def from_tuple(cls, result: tuple[bool, Any | str]) -> Self:
        """Create a GuardrailResult from a validation tuple.

        Args:
            result: A tuple of (success, data) where data is either the validated result or error message.

        Returns:
            A new instance with the tuple data.
        """
        success, data = result
        return cls(
            success=success,
            result=data if success else None,
            error=data if not success else None,
        )


def process_guardrail(
    output: TaskOutput | LiteAgentOutput,
    guardrail: Callable[[Any], tuple[bool, Any | str]],
    retry_count: int,
    event_source: Any | None = None,
) -> GuardrailResult:
    """Process the guardrail for the agent output.

    Args:
        output: The output to validate with the guardrail
        guardrail: The guardrail to validate the output with
        retry_count: The number of times the guardrail has been retried
        event_source: The source of the guardrail to be sent in events

    Returns:
        GuardrailResult: The result of the guardrail validation

    Raises:
        TypeError: If output is not a TaskOutput or LiteAgentOutput
        ValueError: If guardrail is None
    """
    from crewai.lite_agent import LiteAgentOutput
    from crewai.tasks.task_output import TaskOutput

    if not isinstance(output, (TaskOutput, LiteAgentOutput)):
        raise TypeError("Output must be a TaskOutput or LiteAgentOutput")
    if guardrail is None:
        raise ValueError("Guardrail must not be None")

    from crewai.events.event_bus import crewai_event_bus
    from crewai.events.types.llm_guardrail_events import (
        LLMGuardrailCompletedEvent,
        LLMGuardrailStartedEvent,
    )

    crewai_event_bus.emit(
        event_source,
        LLMGuardrailStartedEvent(guardrail=guardrail, retry_count=retry_count),
    )

    result = guardrail(output)
    guardrail_result = GuardrailResult.from_tuple(result)

    crewai_event_bus.emit(
        event_source,
        LLMGuardrailCompletedEvent(
            success=guardrail_result.success,
            result=guardrail_result.result,
            error=guardrail_result.error,
            retry_count=retry_count,
        ),
    )

    return guardrail_result
