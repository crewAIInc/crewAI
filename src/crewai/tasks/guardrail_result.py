"""
Module for handling task guardrail validation results.

This module provides the GuardrailResult class which standardizes
the way task guardrails return their validation results.
"""

from typing import Any, Optional, Tuple, Union
from pydantic import BaseModel


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
