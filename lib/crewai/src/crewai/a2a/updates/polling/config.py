"""Polling update mechanism configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PollingConfig(BaseModel):
    """Configuration for polling-based task updates.

    Attributes:
        interval: Seconds between poll attempts.
        timeout: Max seconds to poll before raising timeout error.
        max_polls: Max number of poll attempts.
        history_length: Number of messages to retrieve per poll.
    """

    interval: float = Field(
        default=2.0, gt=0, description="Seconds between poll attempts"
    )
    timeout: float | None = Field(default=None, gt=0, description="Max seconds to poll")
    max_polls: int | None = Field(default=None, gt=0, description="Max poll attempts")
    history_length: int = Field(
        default=100, gt=0, description="Messages to retrieve per poll"
    )
