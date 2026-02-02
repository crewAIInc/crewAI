"""Transport negotiation utilities for A2A protocol.

This module provides functionality for negotiating the transport protocol
between an A2A client and server based on their respective capabilities
and preferences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from a2a.types import AgentCard, AgentInterface

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.a2a_events import A2ATransportNegotiatedEvent


TransportProtocol = Literal["JSONRPC", "GRPC", "HTTP+JSON"]
NegotiationSource = Literal["client_preferred", "server_preferred", "fallback"]

JSONRPC_TRANSPORT: Literal["JSONRPC"] = "JSONRPC"
GRPC_TRANSPORT: Literal["GRPC"] = "GRPC"
HTTP_JSON_TRANSPORT: Literal["HTTP+JSON"] = "HTTP+JSON"

DEFAULT_TRANSPORT_PREFERENCE: Final[list[TransportProtocol]] = [
    JSONRPC_TRANSPORT,
    GRPC_TRANSPORT,
    HTTP_JSON_TRANSPORT,
]


@dataclass
class NegotiatedTransport:
    """Result of transport negotiation.

    Attributes:
        transport: The negotiated transport protocol.
        url: The URL to use for this transport.
        source: How the transport was selected ('preferred', 'additional', 'fallback').
    """

    transport: str
    url: str
    source: NegotiationSource


class TransportNegotiationError(Exception):
    """Raised when no compatible transport can be negotiated."""

    def __init__(
        self,
        client_transports: list[str],
        server_transports: list[str],
        message: str | None = None,
    ) -> None:
        """Initialize the error with negotiation details.

        Args:
            client_transports: Transports supported by the client.
            server_transports: Transports supported by the server.
            message: Optional custom error message.
        """
        self.client_transports = client_transports
        self.server_transports = server_transports
        if message is None:
            message = (
                f"No compatible transport found. "
                f"Client supports: {client_transports}. "
                f"Server supports: {server_transports}."
            )
        super().__init__(message)


def _get_server_interfaces(agent_card: AgentCard) -> list[AgentInterface]:
    """Extract all available interfaces from an AgentCard.

    Creates a unified list of interfaces including the primary URL and
    any additional interfaces declared by the agent.

    Args:
        agent_card: The agent's card containing transport information.

    Returns:
        List of AgentInterface objects representing all available endpoints.
    """
    interfaces: list[AgentInterface] = []

    primary_transport = agent_card.preferred_transport or JSONRPC_TRANSPORT
    interfaces.append(
        AgentInterface(
            transport=primary_transport,
            url=agent_card.url,
        )
    )

    if agent_card.additional_interfaces:
        for interface in agent_card.additional_interfaces:
            is_duplicate = any(
                i.url == interface.url and i.transport == interface.transport
                for i in interfaces
            )
            if not is_duplicate:
                interfaces.append(interface)

    return interfaces


def negotiate_transport(
    agent_card: AgentCard,
    client_supported_transports: list[str] | None = None,
    client_preferred_transport: str | None = None,
    emit_event: bool = True,
    endpoint: str | None = None,
    a2a_agent_name: str | None = None,
) -> NegotiatedTransport:
    """Negotiate the transport protocol between client and server.

    Compares the client's supported transports with the server's available
    interfaces to find a compatible transport and URL.

    Negotiation logic:
    1. If client_preferred_transport is set and server supports it → use it
    2. Otherwise, if server's preferred is in client's supported → use server's
    3. Otherwise, find first match from client's supported in server's interfaces

    Args:
        agent_card: The server's AgentCard with transport information.
        client_supported_transports: Transports the client can use.
            Defaults to ["JSONRPC"] if not specified.
        client_preferred_transport: Client's preferred transport. If set and
            server supports it, takes priority over server preference.
        emit_event: Whether to emit a transport negotiation event.
        endpoint: Original endpoint URL for event metadata.
        a2a_agent_name: Agent name for event metadata.

    Returns:
        NegotiatedTransport with the selected transport, URL, and source.

    Raises:
        TransportNegotiationError: If no compatible transport is found.
    """
    if client_supported_transports is None:
        client_supported_transports = [JSONRPC_TRANSPORT]

    client_transports = [t.upper() for t in client_supported_transports]
    client_preferred = (
        client_preferred_transport.upper() if client_preferred_transport else None
    )

    server_interfaces = _get_server_interfaces(agent_card)
    server_transports = [i.transport.upper() for i in server_interfaces]

    transport_to_interface: dict[str, AgentInterface] = {}
    for interface in server_interfaces:
        transport_upper = interface.transport.upper()
        if transport_upper not in transport_to_interface:
            transport_to_interface[transport_upper] = interface

    result: NegotiatedTransport | None = None

    if client_preferred and client_preferred in transport_to_interface:
        interface = transport_to_interface[client_preferred]
        result = NegotiatedTransport(
            transport=interface.transport,
            url=interface.url,
            source="client_preferred",
        )
    else:
        server_preferred = (agent_card.preferred_transport or JSONRPC_TRANSPORT).upper()
        if (
            server_preferred in client_transports
            and server_preferred in transport_to_interface
        ):
            interface = transport_to_interface[server_preferred]
            result = NegotiatedTransport(
                transport=interface.transport,
                url=interface.url,
                source="server_preferred",
            )
        else:
            for transport in client_transports:
                if transport in transport_to_interface:
                    interface = transport_to_interface[transport]
                    result = NegotiatedTransport(
                        transport=interface.transport,
                        url=interface.url,
                        source="fallback",
                    )
                    break

    if result is None:
        raise TransportNegotiationError(
            client_transports=client_transports,
            server_transports=server_transports,
        )

    if emit_event:
        crewai_event_bus.emit(
            None,
            A2ATransportNegotiatedEvent(
                endpoint=endpoint or agent_card.url,
                a2a_agent_name=a2a_agent_name or agent_card.name,
                negotiated_transport=result.transport,
                negotiated_url=result.url,
                source=result.source,
                client_supported_transports=client_transports,
                server_supported_transports=server_transports,
                server_preferred_transport=agent_card.preferred_transport
                or JSONRPC_TRANSPORT,
                client_preferred_transport=client_preferred,
            ),
        )

    return result
