"""
Module for handling task guardrail validation results.

This module provides the GuardrailResult class which standardizes
the way task guardrails return their validation results.
"""

from typing import Any, Optional, Tuple, Union

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
