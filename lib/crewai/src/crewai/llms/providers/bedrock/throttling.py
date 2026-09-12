from __future__ import annotations

from typing import Any, Final


BEDROCK_THROTTLING_ERROR_CODES: Final[frozenset[str]] = frozenset(
    {
        "ThrottlingException",
        "RequestLimitExceeded",
        "TooManyRequestsException",
        "ModelNotReadyException",
    }
)

BEDROCK_THROTTLING_ERROR_MESSAGES: Final[tuple[str, ...]] = (
    "throttlingexception",
    "too many tokens, please wait before trying again",
    "too many requests, please wait before trying again",
    "too many requests",
    "rate exceeded",
    "rate limit exceeded",
    "throughput limit exceeded",
    "service quota exceeded",
    "please wait before trying again",
)


def is_bedrock_throttling_error(exception: Exception | Any) -> bool:
    """Check if an exception is an AWS Bedrock throttling / rate-limit response.

    Detects both botocore ClientError response codes and message patterns
    such as 'ThrottlingException' and 'Too many tokens, please wait before trying again'.

    Args:
        exception: The exception or error to check.

    Returns:
        bool: True if the exception represents a retryable throttling response, False otherwise.
    """
    if not isinstance(exception, Exception):
        return False

    # Check botocore ClientError response structure if present
    response = getattr(exception, "response", None)
    if isinstance(response, dict):
        error_info = response.get("Error", {})
        code = error_info.get("Code", "")
        if code in BEDROCK_THROTTLING_ERROR_CODES:
            return True
        msg = str(error_info.get("Message", "")).lower()
        if any(pattern in msg for pattern in BEDROCK_THROTTLING_ERROR_MESSAGES):
            return True

    # Check string representation of the exception
    err_str = str(exception).lower()
    return any(pattern in err_str for pattern in BEDROCK_THROTTLING_ERROR_MESSAGES)
