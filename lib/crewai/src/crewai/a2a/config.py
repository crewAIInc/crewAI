"""A2A configuration types.

This module is separate from experimental.a2a to avoid circular imports.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class A2AConfig(BaseModel):
    """Configuration for A2A protocol integration."""

    endpoint: str = Field(description="A2A agent endpoint URL")
    auth: Any = Field(
        default=None,
        description="Authentication scheme (Bearer, OAuth2, API Key, HTTP Basic/Digest)",
    )
    timeout: int = Field(default=120, description="Request timeout in seconds")
    max_turns: int = Field(
        default=10, description="Maximum conversation turns with A2A agent"
    )
