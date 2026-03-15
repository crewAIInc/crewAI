"""Streaming update mechanism configuration."""

from __future__ import annotations

from pydantic import BaseModel


class StreamingConfig(BaseModel):
    """Configuration for SSE-based task updates."""
