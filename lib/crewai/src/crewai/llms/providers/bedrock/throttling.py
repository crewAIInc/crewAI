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

    Inspects AWS botocore error codes and messages across the exception, its
    `__cause__`, and its `__context__` before falling back to string pattern matching.
    Detects codes like 'ThrottlingException', 'RequestLimitExceeded',
    'TooManyRequestsException', 'ModelNotReadyException', and message patterns
    such as 'too many tokens, please wait before trying again'.

    Args:
        exception: The exception or error to check.

    Returns:
        bool: True if the exception represents a retryable throttling response, False otherwise.
    """
    if not isinstance(exception, BaseException):
        return False

    error_codes_lower = {code.lower() for code in BEDROCK_THROTTLING_ERROR_CODES}

    # First pass: check structured botocore ClientError / EventStreamError codes and messages
    # across the causal exception chain before falling back to string patterns.
    curr: BaseException | None = exception
    visited: set[int] = set()
    while curr is not None and id(curr) not in visited:
        visited.add(id(curr))
        response = getattr(curr, "response", None)
        if isinstance(response, dict):
            error_info = response.get("Error", {})
            if isinstance(error_info, dict):
                code = str(error_info.get("Code", "")).lower()
                if code in error_codes_lower:
                    return True
                msg = str(error_info.get("Message", "")).lower()
                if any(pattern in msg for pattern in BEDROCK_THROTTLING_ERROR_MESSAGES):
                    return True
        curr = curr.__cause__ or curr.__context__

    # Second pass: check string representations across the exception chain
    curr = exception
    visited.clear()
    while curr is not None and id(curr) not in visited:
        visited.add(id(curr))
        err_str = str(curr).lower()
        if any(pattern in err_str for pattern in BEDROCK_THROTTLING_ERROR_MESSAGES):
            return True
        curr = curr.__cause__ or curr.__context__

    return False
