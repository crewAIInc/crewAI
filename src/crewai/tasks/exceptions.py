"""
Module for task-related exceptions.

This module provides custom exceptions used throughout the task system
to provide more specific error handling and context.
"""

from typing import Any, Dict, Optional


class GuardrailValidationError(Exception):
    """Exception raised for guardrail validation errors.
    
    This exception provides detailed context about why a guardrail
    validation failed, including the specific validation that failed
    and any relevant context information.

    Attributes:
        message: A clear description of the validation error
        context: Optional dictionary containing additional error context
    """
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        self.message = message
        self.context = context or {}
        super().__init__(self.message)
