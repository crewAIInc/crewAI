"""Exception class for LLM-related errors."""
from typing import Optional


class LLMError(Exception):
    """Base exception class for LLM operation errors."""

    def __init__(self, message: str, original_error: Optional[Exception] = None):
        """Initialize the LLM error.

        Args:
            message: The error message to display
            original_error: The original exception that caused this error, if any
        """
        super().__init__(message)
        self.original_error = original_error
