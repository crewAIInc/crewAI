"""Usage metrics tracking for CrewAI execution.

This module provides models for tracking token usage and request metrics
during crew and agent execution.
"""

from typing import Any

from pydantic import BaseModel, Field
from typing_extensions import Self


def _coerce_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _first_int(usage_data: dict[str, Any], *keys: str) -> int:
    """Return the first integer-coercible value from ``usage_data`` under any
    of ``keys``. Falls back to ``0`` when nothing matches."""
    for key in keys:
        coerced = _coerce_int(usage_data.get(key))
        if coerced:
            return coerced
    return 0


class UsageMetrics(BaseModel):
    """Track usage metrics for crew execution.

    Attributes:
        total_tokens: Total number of tokens used.
        prompt_tokens: Number of tokens used in prompts.
        cached_prompt_tokens: Number of cached prompt tokens used.
        completion_tokens: Number of tokens used in completions.
        successful_requests: Number of successful requests made.
    """

    total_tokens: int = Field(default=0, description="Total number of tokens used.")
    prompt_tokens: int = Field(
        default=0, description="Number of tokens used in prompts."
    )
    cached_prompt_tokens: int = Field(
        default=0, description="Number of cached prompt tokens used."
    )
    completion_tokens: int = Field(
        default=0, description="Number of tokens used in completions."
    )
    reasoning_tokens: int = Field(
        default=0,
        description="Number of reasoning/thinking tokens (e.g. OpenAI o-series, Gemini thinking).",
    )
    cache_creation_tokens: int = Field(
        default=0,
        description="Number of cache creation tokens (e.g. Anthropic cache writes).",
    )
    successful_requests: int = Field(
        default=0, description="Number of successful requests made."
    )

    def add_usage_metrics(self, usage_metrics: Self) -> None:
        """Add usage metrics from another UsageMetrics object.

        Args:
            usage_metrics: The usage metrics to add.
        """
        self.total_tokens += usage_metrics.total_tokens
        self.prompt_tokens += usage_metrics.prompt_tokens
        self.cached_prompt_tokens += usage_metrics.cached_prompt_tokens
        self.completion_tokens += usage_metrics.completion_tokens
        self.reasoning_tokens += usage_metrics.reasoning_tokens
        self.cache_creation_tokens += usage_metrics.cache_creation_tokens
        self.successful_requests += usage_metrics.successful_requests

    @classmethod
    def from_provider_dict(cls, usage_data: dict[str, Any] | None) -> Self | None:
        """Normalize a provider's raw usage dict into a ``UsageMetrics``.

        Accepts the full set of key aliases CrewAI providers emit:
        ``prompt_tokens`` / ``prompt_token_count`` (Gemini) / ``input_tokens``
        (Anthropic), and the equivalent completion / cached-prompt aliases.
        Mirrors ``BaseLLM._track_token_usage_internal`` so per-LLM totals,
        flow-level aggregation, and OTel spans agree on every provider.

        Returns ``None`` for missing/empty input so callers can decide
        whether to skip the event entirely or treat it as a zero-token
        successful request.
        """
        if not usage_data:
            return None

        prompt_tokens = _first_int(
            usage_data, "prompt_tokens", "prompt_token_count", "input_tokens"
        )
        completion_tokens = _first_int(
            usage_data,
            "completion_tokens",
            "candidates_token_count",
            "output_tokens",
        )
        cached_prompt_tokens = _first_int(
            usage_data,
            "cached_tokens",
            "cached_prompt_tokens",
            "cache_read_input_tokens",
        )
        if not cached_prompt_tokens:
            details = usage_data.get("prompt_tokens_details")
            if isinstance(details, dict):
                cached_prompt_tokens = _coerce_int(details.get("cached_tokens"))

        return cls(
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_prompt_tokens=cached_prompt_tokens,
            reasoning_tokens=_coerce_int(usage_data.get("reasoning_tokens")),
            cache_creation_tokens=_coerce_int(usage_data.get("cache_creation_tokens")),
            successful_requests=1,
        )
