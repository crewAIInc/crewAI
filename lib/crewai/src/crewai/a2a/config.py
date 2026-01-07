"""A2A configuration types.

This module is separate from experimental.a2a to avoid circular imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

if TYPE_CHECKING:
    from a2a.types import TransportProtocol
else:
    # At runtime, use Any as fallback when a2a-sdk is not installed
    # The actual import will happen in _default_transport_protocol() when needed
    TransportProtocol = Any  # type: ignore[misc,assignment]
from typing import Annotated, Any, ClassVar

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    HttpUrl,
    TypeAdapter,
)

from crewai.a2a.auth.schemas import AuthScheme


def _default_transport_protocol() -> Any:
    """Get default transport protocol, handling optional a2a-sdk dependency."""
    try:
        from a2a.types import TransportProtocol

        return TransportProtocol("JSONRPC")
    except ImportError:
        # Fallback to string when a2a-sdk is not installed
        # This will be validated when A2A functionality is actually used
        return "JSONRPC"
try:
    from crewai.a2a.updates import UpdateConfig
except ImportError:
    UpdateConfig = Any  # type: ignore[misc,assignment]


http_url_adapter = TypeAdapter(HttpUrl)

Url = Annotated[
    str,
    BeforeValidator(
        lambda value: str(http_url_adapter.validate_python(value, strict=True))
    ),
]


def _get_default_update_config() -> UpdateConfig:
    from crewai.a2a.updates import StreamingConfig

    return StreamingConfig()


class A2AConfig(BaseModel):
    """Configuration for A2A protocol integration.

    Attributes:
        endpoint: A2A agent endpoint URL.
        auth: Authentication scheme.
        timeout: Request timeout in seconds.
        max_turns: Maximum conversation turns with A2A agent.
        response_model: Optional Pydantic model for structured A2A agent responses.
        fail_fast: If True, raise error when agent unreachable; if False, skip and continue.
        trust_remote_completion_status: If True, return A2A agent's result directly when completed.
        updates: Update mechanism config.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    endpoint: Url = Field(description="A2A agent endpoint URL")
    auth: AuthScheme | None = Field(
        default=None,
        description="Authentication scheme",
    )
    timeout: int = Field(default=120, description="Request timeout in seconds")
    max_turns: int = Field(
        default=10, description="Maximum conversation turns with A2A agent"
    )
    response_model: type[BaseModel] | None = Field(
        default=None,
        description="Optional Pydantic model for structured A2A agent responses",
    )
    fail_fast: bool = Field(
        default=True,
        description="If True, raise error when agent unreachable; if False, skip",
    )
    trust_remote_completion_status: bool = Field(
        default=False,
        description="If True, return A2A result directly when completed",
    )
    updates: UpdateConfig = Field(
        default_factory=_get_default_update_config,
        description="Update mechanism config",
    )
    transport_protocol: TransportProtocol = Field(
        default_factory=_default_transport_protocol,
        description="Optional A2A transport protocol (grpc, jsonrpc, http+json)",
    )
