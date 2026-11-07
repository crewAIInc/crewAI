"""Shared primitives for retrying transient LLM rate limits."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterator, Mapping
import contextvars
import random
import time
from typing import Any, Final, TypeVar, cast


_RETRYABLE_ERROR_CODES: Final[frozenset[str]] = frozenset(
    {
        "429",
        "ratelimiterror",
        "ratelimitexceeded",
        "resourceexhausted",
        "throttlingexception",
        "toomanyrequests",
    }
)
_NON_RETRYABLE_ERROR_CODES: Final[frozenset[str]] = frozenset(
    {
        "servicequotaexceededexception",
    }
)
_RETRYABLE_MESSAGE_MARKERS: Final[tuple[str, ...]] = (
    "rate limit",
    "rate-limit",
    "too many requests",
    "throttled",
    "resource exhausted",
)
_LLM_RATE_LIMIT_MAX_ATTEMPTS: Final = 3
_LLM_RATE_LIMIT_INITIAL_DELAY_SECONDS: Final = 1.0
_LLM_RATE_LIMIT_MAX_DELAY_SECONDS: Final = 8.0
_LLM_RATE_LIMIT_JITTER_RATIO: Final = 0.2
_T = TypeVar("_T")
_active_llm_rate_limit_retry: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_active_llm_rate_limit_retry", default=False
)


class _ThrottlingErrorClassifier:
    """Classify provider exceptions without exposing provider SDK details."""

    @classmethod
    def is_throttling_error(cls, error: BaseException) -> bool:
        """Return whether an error chain represents a transient provider throttle."""
        error_chain = tuple(cls.iter_error_chain(error))
        error_codes = tuple(cls.error_code(candidate) for candidate in error_chain)
        if any(code in _NON_RETRYABLE_ERROR_CODES for code in error_codes):
            return False

        for candidate, error_code in zip(error_chain, error_codes, strict=True):
            if (
                cls.status_code(candidate) == 429
                or error_code in _RETRYABLE_ERROR_CODES
            ):
                return True

        return any(
            marker in str(candidate).lower()
            for candidate in error_chain
            for marker in _RETRYABLE_MESSAGE_MARKERS
        )

    @staticmethod
    def iter_error_chain(error: BaseException) -> Iterator[BaseException]:
        """Yield an exception and its explicit or implicit causes once each."""
        seen: set[int] = set()
        current: BaseException | None = error
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            yield current
            current = current.__cause__ or current.__context__

    @staticmethod
    def error_code(error: BaseException) -> str:
        """Extract and normalize provider error codes across common SDK shapes."""
        code: Any = getattr(error, "code", None)
        response = getattr(error, "response", None)
        if isinstance(response, dict):
            response_error = response.get("Error") or response.get("error") or {}
            if isinstance(response_error, dict):
                code = response_error.get("Code") or response_error.get("code") or code
        return str(code or error.__class__.__name__).replace("_", "").lower()

    @staticmethod
    def status_code(error: BaseException) -> int | None:
        """Extract an HTTP status code when an SDK exposes one."""
        status_code = getattr(error, "status_code", None)
        response = getattr(error, "response", None)
        if status_code is None:
            status_code = getattr(response, "status_code", None)
        return status_code if isinstance(status_code, int) else None


def get_retry_delay_seconds(
    retry_number: int,
    *,
    retry_after_seconds: float | None = None,
    random_value: Callable[[], float] = random.random,
) -> float:
    """Calculate a jittered backoff delay for a one-based retry number.

    A provider-provided retry delay takes precedence over locally calculated
    backoff. ``random_value`` is injectable to make callers' tests deterministic.
    """
    if retry_number < 1:
        raise ValueError("retry_number must be at least 1")
    if retry_after_seconds is not None:
        return max(0.0, retry_after_seconds)

    delay = min(
        _LLM_RATE_LIMIT_INITIAL_DELAY_SECONDS * (2 ** (retry_number - 1)),
        _LLM_RATE_LIMIT_MAX_DELAY_SECONDS,
    )
    jitter = (float(random_value()) * 2 - 1) * _LLM_RATE_LIMIT_JITTER_RATIO
    return float(delay * (1 + jitter))


def run_with_rate_limit_retry(
    operation: Callable[[], _T],
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> _T:
    """Run an operation with retries for transient rate-limit errors only."""

    def attempt() -> tuple[bool, _T | None, Exception | None]:
        """Run one operation while retaining its retryable error."""
        try:
            return True, operation(), None
        except Exception as error:
            return False, None, error

    if _active_llm_rate_limit_retry.get():
        return operation()

    token = _active_llm_rate_limit_retry.set(True)
    try:
        for attempt_number in range(1, _LLM_RATE_LIMIT_MAX_ATTEMPTS + 1):
            succeeded, result, error = attempt()
            if succeeded:
                return cast(_T, result)
            if error is None:
                raise RuntimeError("failed retry attempt did not provide an error")
            if (
                attempt_number == _LLM_RATE_LIMIT_MAX_ATTEMPTS
                or not _ThrottlingErrorClassifier.is_throttling_error(error)
            ):
                raise error
            sleep(
                get_retry_delay_seconds(
                    retry_number=attempt_number,
                    retry_after_seconds=_retry_after_seconds(error),
                )
            )
    finally:
        _active_llm_rate_limit_retry.reset(token)

    raise RuntimeError("rate-limit retry loop completed without a result")


async def arun_with_rate_limit_retry(
    operation: Callable[[], Awaitable[_T]],
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> _T:
    """Asynchronously run an operation with retries for transient rate limits."""

    async def attempt() -> tuple[bool, _T | None, Exception | None]:
        """Run one operation while retaining its retryable error."""
        try:
            return True, await operation(), None
        except Exception as error:
            return False, None, error

    if _active_llm_rate_limit_retry.get():
        return await operation()

    token = _active_llm_rate_limit_retry.set(True)
    try:
        for attempt_number in range(1, _LLM_RATE_LIMIT_MAX_ATTEMPTS + 1):
            succeeded, result, error = await attempt()
            if succeeded:
                return cast(_T, result)
            if error is None:
                raise RuntimeError("failed retry attempt did not provide an error")
            if (
                attempt_number == _LLM_RATE_LIMIT_MAX_ATTEMPTS
                or not _ThrottlingErrorClassifier.is_throttling_error(error)
            ):
                raise error
            await sleep(
                get_retry_delay_seconds(
                    retry_number=attempt_number,
                    retry_after_seconds=_retry_after_seconds(error),
                )
            )
    finally:
        _active_llm_rate_limit_retry.reset(token)

    raise RuntimeError("rate-limit retry loop completed without a result")


def _retry_after_seconds(error: BaseException) -> float | None:
    """Read a numeric ``Retry-After`` hint from common SDK response shapes."""
    for candidate in _ThrottlingErrorClassifier.iter_error_chain(error):
        headers = getattr(candidate, "headers", None)
        response = getattr(candidate, "response", None)
        if headers is None and isinstance(response, dict):
            metadata = response.get("ResponseMetadata") or {}
            headers = metadata.get("HTTPHeaders") or response.get("headers")
        if headers is None:
            headers = getattr(response, "headers", None)
        if not isinstance(headers, Mapping):
            continue
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if retry_after is None:
            continue
        try:
            return max(0.0, float(retry_after))
        except (TypeError, ValueError):
            continue
    return None
