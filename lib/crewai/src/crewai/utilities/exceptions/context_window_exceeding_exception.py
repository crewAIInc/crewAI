from typing import Final

CONTEXT_LIMIT_ERRORS: Final[list[str]] = [
    "expected a string with maximum length",
    "maximum context length",
    "context length exceeded",
    "context_length_exceeded",
    "context window full",
    "too many tokens",
    "input is too long",
    "exceeds token limit",
]


class LLMContextLengthExceededError(Exception):
    """Exception raised when the context length of a language model is exceeded.

    Attributes:
        original_error_message: The original error message from the LLM.
    """

    def __init__(self, error_message: str) -> None:
        """Initialize the exception with the original error message.

        Args:
            error_message: The original error message from the LLM.
        """
        self.original_error_message = error_message
        super().__init__(self._get_error_message(error_message))

    @staticmethod
    def _is_context_limit_error(error_message: str) -> bool:
        """Check if the error message indicates a context length limit error.

        Args:
            error_message: The error message to check.

        Returns:
            True if the error message indicates a context length limit error, False otherwise.
        """
        return any(
            phrase.lower() in error_message.lower() for phrase in CONTEXT_LIMIT_ERRORS
        )

    @staticmethod
    def _get_error_message(error_message: str) -> str:
        """Generate a user-friendly error message based on the original error message.

        Args:
            error_message: The original error message from the LLM.

        Returns:
            A user-friendly error message.
        """
        return (
            f"LLM context length exceeded. Original error: {error_message}\n"
            "Consider using a smaller input or implementing a text splitting strategy."
        )
