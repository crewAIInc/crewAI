"""Configuration for Joy Trust Network integration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal


# Joy's recommended thresholds based on network statistics
# Network average: ~1.1, Network max: ~2.4
RECOMMENDED_THRESHOLDS = {
    "permissive": 1.0,   # Allow most agents, good for low-risk tasks
    "standard": 1.5,     # Reasonable default for general use
    "moderate": 2.0,     # More selective, established agents only
    "strict": 2.5,       # High security, top-tier agents only
}


@dataclass
class JoyTrustConfig:
    """Configuration for Joy Trust Network integration.

    Attributes:
        api_key: Joy Trust API key (optional, increases rate limits)
        min_score: Minimum trust score threshold (0.0 - 5.0)
        fail_open: If True, allow delegation on network errors (default: False)
        timeout: Request timeout in seconds
        cache_ttl: How long to cache trust scores (seconds)
        api_url: Joy Trust API endpoint
    """

    api_key: str | None = field(
        default_factory=lambda: os.getenv("JOY_TRUST_API_KEY")
    )
    min_score: float = field(
        default_factory=lambda: float(os.getenv("JOY_TRUST_MIN_SCORE", "1.5"))
    )
    fail_open: bool = field(
        default_factory=lambda: os.getenv("JOY_TRUST_FAIL_OPEN", "false").lower() == "true"
    )
    timeout: float = 10.0
    cache_ttl: int = 300  # 5 minutes
    api_url: str = "https://joy-connect.fly.dev"

    @classmethod
    def from_env(cls) -> JoyTrustConfig:
        """Create configuration from environment variables."""
        return cls()

    @classmethod
    def with_threshold(
        cls,
        level: Literal["permissive", "standard", "moderate", "strict"],
        **kwargs,
    ) -> JoyTrustConfig:
        """Create configuration with a named threshold level.

        Args:
            level: One of "permissive", "standard", "moderate", "strict"
            **kwargs: Additional configuration options

        Returns:
            JoyTrustConfig with the specified threshold
        """
        min_score = RECOMMENDED_THRESHOLDS.get(level, 1.5)
        return cls(min_score=min_score, **kwargs)
