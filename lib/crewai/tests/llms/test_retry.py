"""Tests for provider-neutral LLM retry policy primitives."""

import pytest

from crewai.llms.retry import (
    DEFAULT_LLM_RETRY_POLICY,
    LLMRetryPolicy,
    get_retry_delay_seconds,
    is_retryable_rate_limit,
)


class _BedrockClientError(Exception):
    def __init__(self, code: str) -> None:
        self.response = {"Error": {"Code": code}}
        super().__init__(code)


@pytest.mark.parametrize(
    "error",
    [
        type("Http429Error", (Exception,), {"status_code": 429})("busy"),
        type("RateLimitError", (Exception,), {})("busy"),
        _BedrockClientError("ThrottlingException"),
        RuntimeError("provider request was throttled"),
    ],
)
def test_is_retryable_rate_limit_recognizes_transient_provider_throttles(
    error: Exception,
) -> None:
    assert is_retryable_rate_limit(error)


def test_is_retryable_rate_limit_follows_exception_causes() -> None:
    try:
        raise _BedrockClientError("ThrottlingException")
    except _BedrockClientError as cause:
        wrapped = RuntimeError("Bedrock request failed")
        wrapped.__cause__ = cause

    assert is_retryable_rate_limit(wrapped)


@pytest.mark.parametrize(
    "error",
    [
        _BedrockClientError("ServiceQuotaExceededException"),
        RuntimeError("too many tokens"),
        ValueError("request validation failed"),
    ],
)
def test_is_retryable_rate_limit_rejects_non_transient_errors(
    error: Exception,
) -> None:
    assert not is_retryable_rate_limit(error)


def test_non_retryable_outer_error_takes_precedence_over_retryable_cause() -> None:
    error = _BedrockClientError("ServiceQuotaExceededException")
    error.__cause__ = _BedrockClientError("ThrottlingException")

    assert not is_retryable_rate_limit(error)


def test_non_retryable_cause_takes_precedence_over_an_ambiguous_wrapper() -> None:
    error = RuntimeError("provider request was throttled")
    error.__cause__ = _BedrockClientError("ServiceQuotaExceededException")

    assert not is_retryable_rate_limit(error)


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
