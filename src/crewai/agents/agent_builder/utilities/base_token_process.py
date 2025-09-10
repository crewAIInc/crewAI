"""Token usage tracking utilities.

This module provides utilities for tracking token consumption and request
metrics during agent execution.
"""

from crewai.types.usage_metrics import UsageMetrics


class TokenProcess:
    """Track token usage during agent processing.

    Attributes:
        total_tokens: Total number of tokens used.
        prompt_tokens: Number of tokens used in prompts.
        cached_prompt_tokens: Number of cached prompt tokens used.
        completion_tokens: Number of tokens used in completions.
        successful_requests: Number of successful requests made.
    """

    def __init__(self) -> None:
        """Initialize token tracking with zero values."""
        self.total_tokens: int = 0
        self.prompt_tokens: int = 0
        self.cached_prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.successful_requests: int = 0

    def sum_prompt_tokens(self, tokens: int) -> None:
        """Add prompt tokens to the running totals.

        Args:
            tokens: Number of prompt tokens to add.
        """
        self.prompt_tokens += tokens
        self.total_tokens += tokens

    def sum_completion_tokens(self, tokens: int) -> None:
        """Add completion tokens to the running totals.

        Args:
            tokens: Number of completion tokens to add.
        """
        self.completion_tokens += tokens
        self.total_tokens += tokens

    def sum_cached_prompt_tokens(self, tokens: int) -> None:
        """Add cached prompt tokens to the running total.

        Args:
            tokens: Number of cached prompt tokens to add.
        """
        self.cached_prompt_tokens += tokens

    def sum_successful_requests(self, requests: int) -> None:
        """Add successful requests to the running total.

        Args:
            requests: Number of successful requests to add.
        """
        self.successful_requests += requests

    def get_summary(self) -> UsageMetrics:
        """Get a summary of all tracked metrics.

        Returns:
            UsageMetrics object with current totals.
        """
        return UsageMetrics(
            total_tokens=self.total_tokens,
            prompt_tokens=self.prompt_tokens,
            cached_prompt_tokens=self.cached_prompt_tokens,
            completion_tokens=self.completion_tokens,
            successful_requests=self.successful_requests,
        )
