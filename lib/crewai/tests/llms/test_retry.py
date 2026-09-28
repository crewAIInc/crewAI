"""Tests for provider-neutral LLM retry behavior."""

from typing import Any

import pytest

from crewai.llms.base_llm import BaseLLM
from crewai.llms.retry import (
    _ThrottlingErrorClassifier,
    arun_with_rate_limit_retry,
    get_retry_delay_seconds,
    run_with_rate_limit_retry,
)


class _StructuredProviderError(Exception):
    def __init__(self, code: str) -> None:
        self.response = {"Error": {"Code": code}}
        super().__init__(code)


class _RetryingLLM(BaseLLM):
    model: str = "test-model"
    outcomes: list[Any]

    def call(self, *args: Any, **kwargs: Any) -> str:
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def acall(self, *args: Any, **kwargs: Any) -> str:
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


@pytest.mark.parametrize(
    "error",
    [
        type("Http429Error", (Exception,), {"status_code": 429})("busy"),
        type("RateLimitError", (Exception,), {})("busy"),
        _StructuredProviderError("TooManyRequests"),
        RuntimeError("provider request was throttled"),
    ],
)
def test_throttling_error_classifier_recognizes_transient_provider_throttles(
    error: Exception,
) -> None:
    assert _ThrottlingErrorClassifier.is_throttling_error(error)


def test_throttling_error_classifier_follows_exception_causes() -> None:
    try:
        raise _StructuredProviderError("TooManyRequests")
    except _StructuredProviderError as cause:
        wrapped = RuntimeError("provider request failed")
        wrapped.__cause__ = cause

    assert _ThrottlingErrorClassifier.is_throttling_error(wrapped)


@pytest.mark.parametrize(
    "error",
    [
        _StructuredProviderError("InvalidRequest"),
        RuntimeError("input exceeds the context window"),
        ValueError("request validation failed"),
    ],
)
def test_throttling_error_classifier_rejects_non_transient_errors(
    error: Exception,
) -> None:
    assert not _ThrottlingErrorClassifier.is_throttling_error(error)


def test_get_retry_delay_seconds_uses_exponential_backoff() -> None:
    assert get_retry_delay_seconds(retry_number=1, random_value=lambda: 0.5) == 1
    assert get_retry_delay_seconds(retry_number=2, random_value=lambda: 0.5) == 2
    assert get_retry_delay_seconds(retry_number=3, random_value=lambda: 0.5) == 4
    assert get_retry_delay_seconds(retry_number=4, random_value=lambda: 0.5) == 8


def test_get_retry_delay_seconds_applies_jitter_and_honors_retry_after() -> None:
    assert get_retry_delay_seconds(retry_number=1, random_value=lambda: 1) == 1.2
    assert get_retry_delay_seconds(retry_number=1, retry_after_seconds=5) == 5


def test_run_with_rate_limit_retry_retries_with_backoff() -> None:
    outcomes: list[str | Exception] = [
        RuntimeError("rate limit exceeded"),
        "complete",
    ]
    delays: list[float] = []

    result = run_with_rate_limit_retry(
        lambda: _pop_outcome(outcomes), sleep=delays.append
    )

    assert result == "complete"
    assert delays == [pytest.approx(1, abs=0.2)]


@pytest.mark.asyncio
async def test_arun_with_rate_limit_retry_retries_with_backoff() -> None:
    outcomes: list[str | Exception] = [
        RuntimeError("rate limit exceeded"),
        "complete",
    ]
    delays: list[float] = []

    async def record_delay(delay: float) -> None:
        delays.append(delay)

    async def operation() -> str:
        return _pop_outcome(outcomes)

    result = await arun_with_rate_limit_retry(operation, sleep=record_delay)

    assert result == "complete"
    assert delays == [pytest.approx(1, abs=0.2)]


def test_base_llm_call_is_automatically_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("crewai.llms.retry.time.sleep", lambda _: None)
    llm = _RetryingLLM(
        model="test-model",
        outcomes=[RuntimeError("rate limit exceeded"), "complete"]
    )

    assert llm.call("hello") == "complete"
    assert llm.outcomes == []


@pytest.mark.asyncio
async def test_base_llm_acall_is_automatically_wrapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("crewai.llms.retry.asyncio.sleep", no_sleep)
    llm = _RetryingLLM(
        model="test-model",
        outcomes=[RuntimeError("rate limit exceeded"), "complete"]
    )

    assert await llm.acall("hello") == "complete"
    assert llm.outcomes == []


def _pop_outcome(outcomes: list[str | Exception]) -> str:
    """Raise a scripted error or return a scripted successful result."""
    outcome = outcomes.pop(0)
    if isinstance(outcome, Exception):
        raise outcome
    return outcome
