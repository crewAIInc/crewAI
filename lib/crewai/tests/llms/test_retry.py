"""Tests for provider-neutral LLM retry behavior."""

from typing import Any

import pytest

from crewai.llms.base_llm import BaseLLM
from crewai.llms.retry import (
    DEFAULT_LLM_RETRY_POLICY,
    LLMRetryPolicy,
    arun_with_rate_limit_retry,
    get_retry_delay_seconds,
    is_throttling_error,
    run_with_rate_limit_retry,
)


class _BedrockClientError(Exception):
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
        _BedrockClientError("ThrottlingException"),
        RuntimeError("provider request was throttled"),
    ],
)
def test_is_throttling_error_recognizes_transient_provider_throttles(
    error: Exception,
) -> None:
    assert is_throttling_error(error)


def test_is_throttling_error_recognizes_bedrock_error_codes() -> None:
    error = _BedrockClientError("ThrottlingException")

    assert is_throttling_error(error)


def test_is_throttling_error_follows_exception_causes() -> None:
    try:
        raise _BedrockClientError("ThrottlingException")
    except _BedrockClientError as cause:
        wrapped = RuntimeError("Bedrock request failed")
        wrapped.__cause__ = cause

    assert is_throttling_error(wrapped)


@pytest.mark.parametrize(
    "error",
    [
        _BedrockClientError("ServiceQuotaExceededException"),
        RuntimeError("too many tokens"),
        ValueError("request validation failed"),
    ],
)
def test_is_throttling_error_rejects_non_transient_errors(
    error: Exception,
) -> None:
    assert not is_throttling_error(error)


def test_non_retryable_outer_error_takes_precedence_over_retryable_cause() -> None:
    error = _BedrockClientError("ServiceQuotaExceededException")
    error.__cause__ = _BedrockClientError("ThrottlingException")

    assert not is_throttling_error(error)


def test_non_retryable_cause_takes_precedence_over_an_ambiguous_wrapper() -> None:
    error = RuntimeError("provider request was throttled")
    error.__cause__ = _BedrockClientError("ServiceQuotaExceededException")

    assert not is_throttling_error(error)


def test_get_retry_delay_seconds_uses_exponential_backoff() -> None:
    policy = LLMRetryPolicy(jitter_ratio=0)

    assert get_retry_delay_seconds(policy, retry_number=1) == 1
    assert get_retry_delay_seconds(policy, retry_number=2) == 2
    assert get_retry_delay_seconds(policy, retry_number=3) == 4
    assert get_retry_delay_seconds(policy, retry_number=4) == 8


def test_get_retry_delay_seconds_applies_jitter_and_honors_retry_after() -> None:
    policy = LLMRetryPolicy(jitter_ratio=0.2)

    assert get_retry_delay_seconds(policy, retry_number=1, random_value=lambda: 1) == 1.2
    assert get_retry_delay_seconds(policy, retry_number=1, retry_after_seconds=5) == 5


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_attempts": 0},
        {"initial_delay_seconds": 0},
        {"max_delay_seconds": 0.5},
        {"jitter_ratio": -0.1},
        {"jitter_ratio": 1.1},
    ],
)
def test_retry_policy_rejects_invalid_settings(kwargs: dict[str, float | int]) -> None:
    with pytest.raises(ValueError):
        LLMRetryPolicy(**kwargs)


def test_default_policy_allows_two_retries_after_the_initial_request() -> None:
    assert DEFAULT_LLM_RETRY_POLICY.max_attempts == 3


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
