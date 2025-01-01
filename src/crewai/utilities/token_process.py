"""Token processing utility for tracking and managing token usage."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from crewai.types.usage_metrics import UsageMetrics


class TokenProcess:
    """Handles token processing and tracking for agents."""
    
    def __init__(self):
        """Initialize the token processor."""
        self._total_tokens = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._cached_prompt_tokens = 0
        self._successful_requests = 0
    
    def sum_prompt_tokens(self, count: int) -> None:
        """Add to prompt token count.
        
        Args:
            count (int): Number of prompt tokens to add
        """
        self._prompt_tokens += count
        self._total_tokens += count
    
    def sum_completion_tokens(self, count: int) -> None:
        """Add to completion token count.
        
        Args:
            count (int): Number of completion tokens to add
        """
        self._completion_tokens += count
        self._total_tokens += count
    
    def sum_cached_prompt_tokens(self, count: int) -> None:
        """Add to cached prompt token count.
        
        Args:
            count (int): Number of cached prompt tokens to add
        """
        self._cached_prompt_tokens += count
    
    def sum_successful_requests(self, count: int) -> None:
        """Add to successful requests count.
        
        Args:
            count (int): Number of successful requests to add
        """
        self._successful_requests += count
    
    def reset(self) -> None:
        """Reset all token counts to zero."""
        self._total_tokens = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._cached_prompt_tokens = 0
        self._successful_requests = 0
        
    def get_summary(self) -> UsageMetrics:
        """Get a summary of token usage.
        
        Returns:
            UsageMetrics: Object containing token usage metrics
        """
        return UsageMetrics(
            total_tokens=self._total_tokens,
            prompt_tokens=self._prompt_tokens,
            cached_prompt_tokens=self._cached_prompt_tokens,
            completion_tokens=self._completion_tokens,
            successful_requests=self._successful_requests
        )
