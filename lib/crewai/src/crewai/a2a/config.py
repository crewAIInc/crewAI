"""A2A configuration types.

This module is separate from experimental.a2a to avoid circular imports.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
    HttpUrl,
    TypeAdapter,
)

from crewai.a2a.auth.schemas import AuthScheme


http_url_adapter = TypeAdapter(HttpUrl)

Url = Annotated[
    str,
    BeforeValidator(
        lambda value: str(http_url_adapter.validate_python(value, strict=True))
    ),
]


class A2AConfig(BaseModel):
    """Configuration for A2A protocol integration.

    Attributes:
        endpoint: A2A agent endpoint URL.
        auth: Authentication scheme (Bearer, OAuth2, API Key, HTTP Basic/Digest).
        timeout: Request timeout in seconds (default: 120).
        max_turns: Maximum conversation turns with A2A agent (default: 10).
        response_model: Optional Pydantic model for structured A2A agent responses.
        fail_fast: If True, raise error when agent unreachable; if False, skip and continue (default: True).
        trust_remote_completion_status: If True, return A2A agent's result directly when status is "completed"; if False, always ask server agent to respond (default: False).
    """

    endpoint: Url = Field(description="A2A agent endpoint URL")
    auth: AuthScheme | None = Field(
        default=None,
        description="Authentication scheme (Bearer, OAuth2, API Key, HTTP Basic/Digest)",
    )
    timeout: int = Field(default=120, description="Request timeout in seconds")
    max_turns: int = Field(
        default=10, description="Maximum conversation turns with A2A agent"
    )
    response_model: type[BaseModel] | None = Field(
        default=None,
        description="Optional Pydantic model for structured A2A agent responses. When specified, the A2A agent is expected to return JSON matching this schema.",
    )
    fail_fast: bool = Field(
        default=True,
        description="If True, raise an error immediately when the A2A agent is unreachable. If False, skip the A2A agent and continue execution.",
    )
    trust_remote_completion_status: bool = Field(
        default=False,
        description='If True, return the A2A agent\'s result directly when status is "completed" without asking the server agent to respond. If False, always ask the server agent to respond, allowing it to potentially delegate again.',
    )
