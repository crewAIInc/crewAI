"""A2A configuration types.

This module is separate from experimental.a2a to avoid circular imports.
"""

from __future__ import annotations

from importlib.metadata import version
from typing import Any, ClassVar, Literal

from a2a.types import (
    AgentCapabilities,
    AgentCardSignature,
    AgentInterface,
    AgentProvider,
    AgentSkill,
    SecurityScheme,
)
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import deprecated

from crewai.a2a.auth.schemas import AuthScheme
from crewai.a2a.types import TransportType, Url


try:
    from crewai.a2a.updates import UpdateConfig
except ImportError:
    UpdateConfig = Any  # type: ignore[misc,assignment]


def _get_default_update_config() -> UpdateConfig:
    from crewai.a2a.updates import StreamingConfig

    return StreamingConfig()


@deprecated(
    """
    `crewai.a2a.config.A2AConfig` is deprecated and will be removed in v2.0.0,
    use `crewai.a2a.config.A2AClientConfig` or `crewai.a2a.config.A2AServerConfig` instead.
    """,
    category=FutureWarning,
)
class A2AConfig(BaseModel):
    """Configuration for A2A protocol integration.

    Deprecated:
        Use A2AClientConfig instead. This class will be removed in a future version.

    Attributes:
        endpoint: A2A agent endpoint URL.
        auth: Authentication scheme.
        timeout: Request timeout in seconds.
        max_turns: Maximum conversation turns with A2A agent.
        response_model: Optional Pydantic model for structured A2A agent responses.
        fail_fast: If True, raise error when agent unreachable; if False, skip and continue.
        trust_remote_completion_status: If True, return A2A agent's result directly when completed.
        updates: Update mechanism config.
        transport_protocol: A2A transport protocol (grpc, jsonrpc, http+json).
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
    transport_protocol: Literal["JSONRPC", "GRPC", "HTTP+JSON"] = Field(
        default="JSONRPC",
        description="Specified mode of A2A transport protocol",
    )


class A2AClientConfig(BaseModel):
    """Configuration for connecting to remote A2A agents.

    Attributes:
        endpoint: A2A agent endpoint URL.
        auth: Authentication scheme.
        timeout: Request timeout in seconds.
        max_turns: Maximum conversation turns with A2A agent.
        response_model: Optional Pydantic model for structured A2A agent responses.
        fail_fast: If True, raise error when agent unreachable; if False, skip and continue.
        trust_remote_completion_status: If True, return A2A agent's result directly when completed.
        updates: Update mechanism config.
        accepted_output_modes: Media types the client can accept in responses.
        supported_transports: Ordered list of transport protocols the client supports.
        use_client_preference: Whether to prioritize client transport preferences over server.
        extensions: Extension URIs the client supports.
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
    accepted_output_modes: list[str] = Field(
        default_factory=lambda: ["application/json"],
        description="Media types the client can accept in responses",
    )
    supported_transports: list[str] = Field(
        default_factory=lambda: ["JSONRPC"],
        description="Ordered list of transport protocols the client supports",
    )
    use_client_preference: bool = Field(
        default=False,
        description="Whether to prioritize client transport preferences over server",
    )
    extensions: list[str] = Field(
        default_factory=list,
        description="Extension URIs the client supports",
    )
    transport_protocol: Literal["JSONRPC", "GRPC", "HTTP+JSON"] = Field(
        default="JSONRPC",
        description="Specified mode of A2A transport protocol",
    )


class A2AServerConfig(BaseModel):
    """Configuration for exposing a Crew or Agent as an A2A server.

    All fields correspond to A2A AgentCard fields. Fields like name, description,
    and skills can be auto-derived from the Crew/Agent if not provided.

    Attributes:
        name: Human-readable name for the agent.
        description: Human-readable description of the agent.
        version: Version string for the agent card.
        skills: List of agent skills/capabilities.
        default_input_modes: Default supported input MIME types.
        default_output_modes: Default supported output MIME types.
        capabilities: Declaration of optional capabilities.
        preferred_transport: Transport protocol for the preferred endpoint.
        protocol_version: A2A protocol version this agent supports.
        provider: Information about the agent's service provider.
        documentation_url: URL to the agent's documentation.
        icon_url: URL to an icon for the agent.
        additional_interfaces: Additional supported interfaces.
        security: Security requirement objects for all interactions.
        security_schemes: Security schemes available to authorize requests.
        supports_authenticated_extended_card: Whether agent provides extended card to authenticated users.
        url: Preferred endpoint URL for the agent.
        signatures: JSON Web Signatures for the AgentCard.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    name: str | None = Field(
        default=None,
        description="Human-readable name for the agent. Auto-derived from Crew/Agent if not provided.",
    )
    description: str | None = Field(
        default=None,
        description="Human-readable description of the agent. Auto-derived from Crew/Agent if not provided.",
    )
    version: str = Field(
        default="1.0.0",
        description="Version string for the agent card",
    )
    skills: list[AgentSkill] = Field(
        default_factory=list,
        description="List of agent skills. Auto-derived from tasks/tools if not provided.",
    )
    default_input_modes: list[str] = Field(
        default_factory=lambda: ["text/plain", "application/json"],
        description="Default supported input MIME types",
    )
    default_output_modes: list[str] = Field(
        default_factory=lambda: ["text/plain", "application/json"],
        description="Default supported output MIME types",
    )
    capabilities: AgentCapabilities = Field(
        default_factory=lambda: AgentCapabilities(
            streaming=True,
            push_notifications=False,
        ),
        description="Declaration of optional capabilities supported by the agent",
    )
    preferred_transport: TransportType = Field(
        default="JSONRPC",
        description="Transport protocol for the preferred endpoint",
    )
    protocol_version: str = Field(
        default_factory=lambda: version("a2a-sdk"),
        description="A2A protocol version this agent supports",
    )
    provider: AgentProvider | None = Field(
        default=None,
        description="Information about the agent's service provider",
    )
    documentation_url: Url | None = Field(
        default=None,
        description="URL to the agent's documentation",
    )
    icon_url: Url | None = Field(
        default=None,
        description="URL to an icon for the agent",
    )
    additional_interfaces: list[AgentInterface] = Field(
        default_factory=list,
        description="Additional supported interfaces (transport and URL combinations)",
    )
    security: list[dict[str, list[str]]] = Field(
        default_factory=list,
        description="Security requirement objects for all agent interactions",
    )
    security_schemes: dict[str, SecurityScheme] = Field(
        default_factory=dict,
        description="Security schemes available to authorize requests",
    )
    supports_authenticated_extended_card: bool = Field(
        default=False,
        description="Whether agent provides extended card to authenticated users",
    )
    url: Url | None = Field(
        default=None,
        description="Preferred endpoint URL for the agent. Set at runtime if not provided.",
    )
    signatures: list[AgentCardSignature] = Field(
        default_factory=list,
        description="JSON Web Signatures for the AgentCard",
    )
