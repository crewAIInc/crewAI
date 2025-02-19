from typing import Optional


class FeedbackProcessingError(Exception):
    """Exception raised when feedback processing fails."""
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        self.original_error = original_error
        super().__init__(message)
