"""A2A configuration types.

This module is separate from experimental.a2a to avoid circular imports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Literal, cast
import warnings

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FilePath,
    PrivateAttr,
    SecretStr,
    model_validator,
)
from typing_extensions import Self, deprecated

from crewai.a2a.auth.client_schemes import ClientAuthScheme
from crewai.a2a.auth.server_schemes import ServerAuthScheme
from crewai.a2a.extensions.base import ValidatedA2AExtension
from crewai.a2a.types import ProtocolVersion, TransportType, Url


try:
    from a2a.types import (
        AgentCapabilities,
        AgentCardSignature,
        AgentInterface,
        AgentProvider,
        AgentSkill,
        SecurityScheme,
    )

    from crewai.a2a.extensions.server import ServerExtension
    from crewai.a2a.updates import UpdateConfig
except ImportError:
    UpdateConfig: Any = Any  # type: ignore[no-redef]
    AgentCapabilities: Any = Any  # type: ignore[no-redef]
    AgentCardSignature: Any = Any  # type: ignore[no-redef]
    AgentInterface: Any = Any  # type: ignore[no-redef]
    AgentProvider: Any = Any  # type: ignore[no-redef]
    SecurityScheme: Any = Any  # type: ignore[no-redef]
    AgentSkill: Any = Any  # type: ignore[no-redef]
    ServerExtension: Any = Any  # type: ignore[no-redef]


def _get_default_update_config() -> UpdateConfig:
    from crewai.a2a.updates import StreamingConfig

    return StreamingConfig()


SigningAlgorithm = Literal[
    "RS256",
    "RS384",
    "RS512",
    "ES256",
    "ES384",
    "ES512",
    "PS256",
    "PS384",
    "PS512",
]


class AgentCardSigningConfig(BaseModel):
    """Configuration for AgentCard JWS signing.

    Provides the private key and algorithm settings for signing AgentCards.
    Either private_key_path or private_key_pem must be provided, but not both.

    Attributes:
        private_key_path: Path to a PEM-encoded private key file.
        private_key_pem: PEM-encoded private key as a secret string.
        key_id: Optional key identifier for the JWS header (kid claim).
        algorithm: Signing algorithm (RS256, ES256, PS256, etc.).
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    private_key_path: FilePath | None = Field(
        default=None,
        description="Path to PEM-encoded private key file",
    )
    private_key_pem: SecretStr | None = Field(
        default=None,
        description="PEM-encoded private key",
    )
    key_id: str | None = Field(
        default=None,
        description="Key identifier for JWS header (kid claim)",
    )
    algorithm: SigningAlgorithm = Field(
        default="RS256",
        description="Signing algorithm (RS256, ES256, PS256, etc.)",
    )

    @model_validator(mode="after")
    def _validate_key_source(self) -> Self:
        """Ensure exactly one key source is provided."""
        has_path = self.private_key_path is not None
        has_pem = self.private_key_pem is not None

        if not has_path and not has_pem:
            raise ValueError(
                "Either private_key_path or private_key_pem must be provided"
            )
        if has_path and has_pem:
            raise ValueError(
                "Only one of private_key_path or private_key_pem should be provided"
            )
        return self

    def get_private_key(self) -> str:
        """Get the private key content.

        Returns:
            The PEM-encoded private key as a string.
        """
        if self.private_key_pem:
            return self.private_key_pem.get_secret_value()
        if self.private_key_path:
            return Path(self.private_key_path).read_text()
        raise ValueError("No private key configured")


class GRPCServerConfig(BaseModel):
    """gRPC server transport configuration.

    Presence of this config in ServerTransportConfig.grpc enables gRPC transport.

    Attributes:
        host: Hostname to advertise in agent cards (default: localhost).
            Use docker service name (e.g., 'web') for docker-compose setups.
        port: Port for the gRPC server.
        tls_cert_path: Path to TLS certificate file for gRPC.
        tls_key_path: Path to TLS private key file for gRPC.
        max_workers: Maximum number of workers for the gRPC thread pool.
        reflection_enabled: Whether to enable gRPC reflection for debugging.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    host: str = Field(
        default="localhost",
        description="Hostname to advertise in agent cards for gRPC connections",
    )
    port: int = Field(
        default=50051,
        description="Port for the gRPC server",
    )
    tls_cert_path: str | None = Field(
        default=None,
        description="Path to TLS certificate file for gRPC",
    )
    tls_key_path: str | None = Field(
        default=None,
        description="Path to TLS private key file for gRPC",
    )
    max_workers: int = Field(
        default=10,
        description="Maximum number of workers for the gRPC thread pool",
    )
    reflection_enabled: bool = Field(
        default=False,
        description="Whether to enable gRPC reflection for debugging",
    )


class GRPCClientConfig(BaseModel):
    """gRPC client transport configuration.

    Attributes:
        max_send_message_length: Maximum size for outgoing messages in bytes.
        max_receive_message_length: Maximum size for incoming messages in bytes.
        keepalive_time_ms: Time between keepalive pings in milliseconds.
        keepalive_timeout_ms: Timeout for keepalive ping response in milliseconds.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    max_send_message_length: int | None = Field(
        default=None,
        description="Maximum size for outgoing messages in bytes",
    )
    max_receive_message_length: int | None = Field(
        default=None,
        description="Maximum size for incoming messages in bytes",
    )
    keepalive_time_ms: int | None = Field(
        default=None,
        description="Time between keepalive pings in milliseconds",
    )
    keepalive_timeout_ms: int | None = Field(
        default=None,
        description="Timeout for keepalive ping response in milliseconds",
    )


class JSONRPCServerConfig(BaseModel):
    """JSON-RPC server transport configuration.

    Presence of this config in ServerTransportConfig.jsonrpc enables JSON-RPC transport.

    Attributes:
        rpc_path: URL path for the JSON-RPC endpoint.
        agent_card_path: URL path for the agent card endpoint.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    rpc_path: str = Field(
        default="/a2a",
        description="URL path for the JSON-RPC endpoint",
    )
    agent_card_path: str = Field(
        default="/.well-known/agent-card.json",
        description="URL path for the agent card endpoint",
    )


class JSONRPCClientConfig(BaseModel):
    """JSON-RPC client transport configuration.

    Attributes:
        max_request_size: Maximum request body size in bytes.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    max_request_size: int | None = Field(
        default=None,
        description="Maximum request body size in bytes",
    )


class HTTPJSONConfig(BaseModel):
    """HTTP+JSON transport configuration.

    Presence of this config in ServerTransportConfig.http_json enables HTTP+JSON transport.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")


class ServerPushNotificationConfig(BaseModel):
    """Configuration for outgoing webhook push notifications.

    Controls how the server signs and delivers push notifications to clients.

    Attributes:
        signature_secret: Shared secret for HMAC-SHA256 signing of outgoing webhooks.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    signature_secret: SecretStr | None = Field(
        default=None,
        description="Shared secret for HMAC-SHA256 signing of outgoing push notifications",
    )


class ServerTransportConfig(BaseModel):
    """Transport configuration for A2A server.

    Groups all transport-related settings including preferred transport
    and protocol-specific configurations.

    Attributes:
        preferred: Transport protocol for the preferred endpoint.
        jsonrpc: JSON-RPC server transport configuration.
        grpc: gRPC server transport configuration.
        http_json: HTTP+JSON transport configuration.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    preferred: TransportType = Field(
        default="JSONRPC",
        description="Transport protocol for the preferred endpoint",
    )
    jsonrpc: JSONRPCServerConfig = Field(
        default_factory=JSONRPCServerConfig,
        description="JSON-RPC server transport configuration",
    )
    grpc: GRPCServerConfig | None = Field(
        default=None,
        description="gRPC server transport configuration",
    )
    http_json: HTTPJSONConfig | None = Field(
        default=None,
        description="HTTP+JSON transport configuration",
    )


def _migrate_client_transport_fields(
    transport: ClientTransportConfig,
    transport_protocol: TransportType | None,
    supported_transports: list[TransportType] | None,
) -> None:
    """Migrate deprecated transport fields to new config."""
    if transport_protocol is not None:
        warnings.warn(
            "transport_protocol is deprecated, use transport=ClientTransportConfig(preferred=...) instead",
            FutureWarning,
            stacklevel=5,
        )
        object.__setattr__(transport, "preferred", transport_protocol)
    if supported_transports is not None:
        warnings.warn(
            "supported_transports is deprecated, use transport=ClientTransportConfig(supported=...) instead",
            FutureWarning,
            stacklevel=5,
        )
        object.__setattr__(transport, "supported", supported_transports)


class ClientTransportConfig(BaseModel):
    """Transport configuration for A2A client.

    Groups all client transport-related settings including preferred transport,
    supported transports for negotiation, and protocol-specific configurations.

    Transport negotiation logic:
    1. If `preferred` is set and server supports it → use client's preferred
    2. Otherwise, if server's preferred is in client's `supported` → use server's preferred
    3. Otherwise, find first match from client's `supported` in server's interfaces

    Attributes:
        preferred: Client's preferred transport. If set, client preference takes priority.
        supported: Transports the client can use, in order of preference.
        jsonrpc: JSON-RPC client transport configuration.
        grpc: gRPC client transport configuration.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    preferred: TransportType | None = Field(
        default=None,
        description="Client's preferred transport. If set, takes priority over server preference.",
    )
    supported: list[TransportType] = Field(
        default_factory=lambda: cast(list[TransportType], ["JSONRPC"]),
        description="Transports the client can use, in order of preference",
    )
    jsonrpc: JSONRPCClientConfig = Field(
        default_factory=JSONRPCClientConfig,
        description="JSON-RPC client transport configuration",
    )
    grpc: GRPCClientConfig = Field(
        default_factory=GRPCClientConfig,
        description="gRPC client transport configuration",
    )


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
        client_extensions: Client-side processing hooks for tool injection and prompt augmentation.
        transport: Transport configuration (preferred, supported transports, gRPC settings).
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    endpoint: Url = Field(description="A2A agent endpoint URL")
    auth: ClientAuthScheme | None = Field(
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
    client_extensions: list[ValidatedA2AExtension] = Field(
        default_factory=list,
        description="Client-side processing hooks for tool injection and prompt augmentation",
    )
    transport: ClientTransportConfig = Field(
        default_factory=ClientTransportConfig,
        description="Transport configuration (preferred, supported transports, gRPC settings)",
    )
    transport_protocol: TransportType | None = Field(
        default=None,
        description="Deprecated: Use transport.preferred instead",
        exclude=True,
    )
    supported_transports: list[TransportType] | None = Field(
        default=None,
        description="Deprecated: Use transport.supported instead",
        exclude=True,
    )
    use_client_preference: bool | None = Field(
        default=None,
        description="Deprecated: Set transport.preferred to enable client preference",
        exclude=True,
    )
    _parallel_delegation: bool = PrivateAttr(default=False)

    @model_validator(mode="after")
    def _migrate_deprecated_transport_fields(self) -> Self:
        """Migrate deprecated transport fields to new config."""
        _migrate_client_transport_fields(
            self.transport, self.transport_protocol, self.supported_transports
        )
        if self.use_client_preference is not None:
            warnings.warn(
                "use_client_preference is deprecated, set transport.preferred to enable client preference",
                FutureWarning,
                stacklevel=4,
            )
            if self.use_client_preference and self.transport.supported:
                object.__setattr__(
                    self.transport, "preferred", self.transport.supported[0]
                )
        return self


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
        extensions: Extension URIs the client supports (A2A protocol extensions).
        client_extensions: Client-side processing hooks for tool injection and prompt augmentation.
        transport: Transport configuration (preferred, supported transports, gRPC settings).
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    endpoint: Url = Field(description="A2A agent endpoint URL")
    auth: ClientAuthScheme | None = Field(
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
    extensions: list[str] = Field(
        default_factory=list,
        description="Extension URIs the client supports",
    )
    client_extensions: list[ValidatedA2AExtension] = Field(
        default_factory=list,
        description="Client-side processing hooks for tool injection and prompt augmentation",
    )
    transport: ClientTransportConfig = Field(
        default_factory=ClientTransportConfig,
        description="Transport configuration (preferred, supported transports, gRPC settings)",
    )
    transport_protocol: TransportType | None = Field(
        default=None,
        description="Deprecated: Use transport.preferred instead",
        exclude=True,
    )
    supported_transports: list[TransportType] | None = Field(
        default=None,
        description="Deprecated: Use transport.supported instead",
        exclude=True,
    )
    _parallel_delegation: bool = PrivateAttr(default=False)

    @model_validator(mode="after")
    def _migrate_deprecated_transport_fields(self) -> Self:
        """Migrate deprecated transport fields to new config."""
        _migrate_client_transport_fields(
            self.transport, self.transport_protocol, self.supported_transports
        )
        return self


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
        protocol_version: A2A protocol version this agent supports.
        provider: Information about the agent's service provider.
        documentation_url: URL to the agent's documentation.
        icon_url: URL to an icon for the agent.
        additional_interfaces: Additional supported interfaces.
        security: Security requirement objects for all interactions.
        security_schemes: Security schemes available to authorize requests.
        supports_authenticated_extended_card: Whether agent provides extended card to authenticated users.
        url: Preferred endpoint URL for the agent.
        signing_config: Configuration for signing the AgentCard with JWS.
        signatures: Deprecated. Pre-computed JWS signatures. Use signing_config instead.
        server_extensions: Server-side A2A protocol extensions with on_request/on_response hooks.
        push_notifications: Configuration for outgoing push notifications.
        transport: Transport configuration (preferred transport, gRPC, REST settings).
        auth: Authentication scheme for A2A endpoints.
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
    protocol_version: ProtocolVersion = Field(
        default="0.3.0",
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
        description="Additional supported interfaces.",
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
    signing_config: AgentCardSigningConfig | None = Field(
        default=None,
        description="Configuration for signing the AgentCard with JWS",
    )
    signatures: list[AgentCardSignature] | None = Field(
        default=None,
        description="Deprecated: Use signing_config instead. Pre-computed JWS signatures for the AgentCard.",
        exclude=True,
        deprecated=True,
    )
    server_extensions: list[ServerExtension] = Field(
        default_factory=list,
        description="Server-side A2A protocol extensions that modify agent behavior",
    )
    push_notifications: ServerPushNotificationConfig | None = Field(
        default=None,
        description="Configuration for outgoing push notifications",
    )
    transport: ServerTransportConfig = Field(
        default_factory=ServerTransportConfig,
        description="Transport configuration (preferred transport, gRPC, REST settings)",
    )
    preferred_transport: TransportType | None = Field(
        default=None,
        description="Deprecated: Use transport.preferred instead",
        exclude=True,
        deprecated=True,
    )
    auth: ServerAuthScheme | None = Field(
        default=None,
        description="Authentication scheme for A2A endpoints. Defaults to SimpleTokenAuth using AUTH_TOKEN env var.",
    )

    @model_validator(mode="after")
    def _migrate_deprecated_fields(self) -> Self:
        """Migrate deprecated fields to new config."""
        if self.preferred_transport is not None:
            warnings.warn(
                "preferred_transport is deprecated, use transport=ServerTransportConfig(preferred=...) instead",
                FutureWarning,
                stacklevel=4,
            )
            object.__setattr__(self.transport, "preferred", self.preferred_transport)
        if self.signatures is not None:
            warnings.warn(
                "signatures is deprecated, use signing_config=AgentCardSigningConfig(...) instead. "
                "The signatures field will be removed in v2.0.0.",
                FutureWarning,
                stacklevel=4,
            )
        return self
