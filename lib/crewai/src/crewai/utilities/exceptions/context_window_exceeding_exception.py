from typing import Any, Final


CONTEXT_LIMIT_ERRORS: Final[list[str]] = [
    "expected a string with maximum length",
    "maximum context length",
    "context length exceeded",
    "context_length_exceeded",
    "context window full",
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


class LLMRateLimitExceededError(Exception):
    """Exception raised when an LLM provider temporarily throttles a request.

    Provider adapters raise this error after preserving the upstream message.
    The shared agent execution loop can then retry the same request without
    treating the failure as a context-window overflow.
    """

    def __init__(self, error_message: str) -> None:
        """Initialize the exception with the original provider error message."""
        self.original_error_message = error_message
        super().__init__(error_message)


def is_rate_limit_exceeded(error: Exception) -> bool:
    """Check whether a provider error represents a transient rate limit.

    Native SDKs expose throttles through different shapes. This keeps the
    detection rules shared while each provider retains its own useful message
    when it raises :class:`LLMRateLimitExceededError`.
    """
    response: Any = getattr(error, "response", None)
    status_code = getattr(error, "status_code", None)
    if status_code is None:
        status_code = getattr(response, "status_code", None)
    if status_code == 429:
        return True

    error_code = getattr(error, "code", None)
    if isinstance(response, dict):
        response_error = response.get("Error") or response.get("error") or {}
        if isinstance(response_error, dict):
            error_code = response_error.get("Code") or response_error.get("code")

    normalized_code = (
        str(error_code or error.__class__.__name__).replace("_", "").lower()
    )
    if normalized_code in {
        "429",
        "ratelimiterror",
        "ratelimitexceeded",
        "throttlingexception",
        "resourceexhausted",
        "toomanyrequests",
    }:
        return True

    message = str(error).lower()
    return any(
        phrase in message
        for phrase in (
            "rate limit",
            "rate-limit",
            "too many requests",
            "throttl",
            "resource exhausted",
        )
    )
