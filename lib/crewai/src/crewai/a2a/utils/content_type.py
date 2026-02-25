"""Content type negotiation for A2A protocol.

This module handles negotiation of input/output MIME types between A2A clients
and servers based on AgentCard capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Final, Literal, cast

from a2a.types import Part

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.a2a_events import A2AContentTypeNegotiatedEvent


if TYPE_CHECKING:
    from a2a.types import AgentCard, AgentSkill


TEXT_PLAIN: Literal["text/plain"] = "text/plain"
APPLICATION_JSON: Literal["application/json"] = "application/json"
IMAGE_PNG: Literal["image/png"] = "image/png"
IMAGE_JPEG: Literal["image/jpeg"] = "image/jpeg"
IMAGE_WILDCARD: Literal["image/*"] = "image/*"
APPLICATION_PDF: Literal["application/pdf"] = "application/pdf"
APPLICATION_OCTET_STREAM: Literal["application/octet-stream"] = (
    "application/octet-stream"
)

DEFAULT_CLIENT_INPUT_MODES: Final[list[Literal["text/plain", "application/json"]]] = [
    TEXT_PLAIN,
    APPLICATION_JSON,
]
DEFAULT_CLIENT_OUTPUT_MODES: Final[list[Literal["text/plain", "application/json"]]] = [
    TEXT_PLAIN,
    APPLICATION_JSON,
]


@dataclass
class NegotiatedContentTypes:
    """Result of content type negotiation."""

    input_modes: Annotated[list[str], "Negotiated input MIME types the client can send"]
    output_modes: Annotated[
        list[str], "Negotiated output MIME types the server will produce"
    ]
    effective_input_modes: Annotated[list[str], "Server's effective input modes"]
    effective_output_modes: Annotated[list[str], "Server's effective output modes"]
    skill_name: Annotated[
        str | None, "Skill name if negotiation was skill-specific"
    ] = None


class ContentTypeNegotiationError(Exception):
    """Raised when no compatible content types can be negotiated."""

    def __init__(
        self,
        client_input_modes: list[str],
        client_output_modes: list[str],
        server_input_modes: list[str],
        server_output_modes: list[str],
        direction: str = "both",
        message: str | None = None,
    ) -> None:
        self.client_input_modes = client_input_modes
        self.client_output_modes = client_output_modes
        self.server_input_modes = server_input_modes
        self.server_output_modes = server_output_modes
        self.direction = direction

        if message is None:
            if direction == "input":
                message = (
                    f"No compatible input content types. "
                    f"Client supports: {client_input_modes}, "
                    f"Server accepts: {server_input_modes}"
                )
            elif direction == "output":
                message = (
                    f"No compatible output content types. "
                    f"Client accepts: {client_output_modes}, "
                    f"Server produces: {server_output_modes}"
                )
            else:
                message = (
                    f"No compatible content types. "
                    f"Input - Client: {client_input_modes}, Server: {server_input_modes}. "
                    f"Output - Client: {client_output_modes}, Server: {server_output_modes}"
                )

        super().__init__(message)


def _normalize_mime_type(mime_type: str) -> str:
    """Normalize MIME type for comparison (lowercase, strip whitespace)."""
    return mime_type.lower().strip()


def _mime_types_compatible(client_type: str, server_type: str) -> bool:
    """Check if two MIME types are compatible.

    Handles wildcards like image/* matching image/png.
    """
    client_normalized = _normalize_mime_type(client_type)
    server_normalized = _normalize_mime_type(server_type)

    if client_normalized == server_normalized:
        return True

    if "*" in client_normalized or "*" in server_normalized:
        client_parts = client_normalized.split("/")
        server_parts = server_normalized.split("/")

        if len(client_parts) == 2 and len(server_parts) == 2:
            type_match = (
                client_parts[0] == server_parts[0]
                or client_parts[0] == "*"
                or server_parts[0] == "*"
            )
            subtype_match = (
                client_parts[1] == server_parts[1]
                or client_parts[1] == "*"
                or server_parts[1] == "*"
            )
            return type_match and subtype_match

    return False


def _find_compatible_modes(
    client_modes: list[str], server_modes: list[str]
) -> list[str]:
    """Find compatible MIME types between client and server.

    Returns modes in client preference order.
    """
    compatible = []
    for client_mode in client_modes:
        for server_mode in server_modes:
            if _mime_types_compatible(client_mode, server_mode):
                if "*" in client_mode and "*" not in server_mode:
                    if server_mode not in compatible:
                        compatible.append(server_mode)
                else:
                    if client_mode not in compatible:
                        compatible.append(client_mode)
                break
    return compatible


def _get_effective_modes(
    agent_card: AgentCard,
    skill_name: str | None = None,
) -> tuple[list[str], list[str], AgentSkill | None]:
    """Get effective input/output modes from agent card.

    If skill_name is provided and the skill has custom modes, those are used.
    Otherwise, falls back to agent card defaults.
    """
    skill: AgentSkill | None = None

    if skill_name and agent_card.skills:
        for s in agent_card.skills:
            if s.name == skill_name or s.id == skill_name:
                skill = s
                break

    if skill:
        input_modes = (
            skill.input_modes if skill.input_modes else agent_card.default_input_modes
        )
        output_modes = (
            skill.output_modes
            if skill.output_modes
            else agent_card.default_output_modes
        )
    else:
        input_modes = agent_card.default_input_modes
        output_modes = agent_card.default_output_modes

    return input_modes, output_modes, skill


def negotiate_content_types(
    agent_card: AgentCard,
    client_input_modes: list[str] | None = None,
    client_output_modes: list[str] | None = None,
    skill_name: str | None = None,
    emit_event: bool = True,
    endpoint: str | None = None,
    a2a_agent_name: str | None = None,
    strict: bool = False,
) -> NegotiatedContentTypes:
    """Negotiate content types between client and server.

    Args:
        agent_card: The remote agent's card with capability info.
        client_input_modes: MIME types the client can send. Defaults to text/plain and application/json.
        client_output_modes: MIME types the client can accept. Defaults to text/plain and application/json.
        skill_name: Optional skill to use for mode lookup.
        emit_event: Whether to emit a content type negotiation event.
        endpoint: Agent endpoint (for event metadata).
        a2a_agent_name: Agent name (for event metadata).
        strict: If True, raises error when no compatible types found.
            If False, returns empty lists for incompatible directions.

    Returns:
        NegotiatedContentTypes with compatible input and output modes.

    Raises:
        ContentTypeNegotiationError: If strict=True and no compatible types found.
    """
    if client_input_modes is None:
        client_input_modes = cast(list[str], DEFAULT_CLIENT_INPUT_MODES.copy())
    if client_output_modes is None:
        client_output_modes = cast(list[str], DEFAULT_CLIENT_OUTPUT_MODES.copy())

    server_input_modes, server_output_modes, skill = _get_effective_modes(
        agent_card, skill_name
    )

    compatible_input = _find_compatible_modes(client_input_modes, server_input_modes)
    compatible_output = _find_compatible_modes(client_output_modes, server_output_modes)

    if strict:
        if not compatible_input and not compatible_output:
            raise ContentTypeNegotiationError(
                client_input_modes=client_input_modes,
                client_output_modes=client_output_modes,
                server_input_modes=server_input_modes,
                server_output_modes=server_output_modes,
            )
        if not compatible_input:
            raise ContentTypeNegotiationError(
                client_input_modes=client_input_modes,
                client_output_modes=client_output_modes,
                server_input_modes=server_input_modes,
                server_output_modes=server_output_modes,
                direction="input",
            )
        if not compatible_output:
            raise ContentTypeNegotiationError(
                client_input_modes=client_input_modes,
                client_output_modes=client_output_modes,
                server_input_modes=server_input_modes,
                server_output_modes=server_output_modes,
                direction="output",
            )

    result = NegotiatedContentTypes(
        input_modes=compatible_input,
        output_modes=compatible_output,
        effective_input_modes=server_input_modes,
        effective_output_modes=server_output_modes,
        skill_name=skill.name if skill else None,
    )

    if emit_event:
        crewai_event_bus.emit(
            None,
            A2AContentTypeNegotiatedEvent(
                endpoint=endpoint or agent_card.url,
                a2a_agent_name=a2a_agent_name or agent_card.name,
                skill_name=skill_name,
                client_input_modes=client_input_modes,
                client_output_modes=client_output_modes,
                server_input_modes=server_input_modes,
                server_output_modes=server_output_modes,
                negotiated_input_modes=compatible_input,
                negotiated_output_modes=compatible_output,
                negotiation_success=bool(compatible_input and compatible_output),
            ),
        )

    return result


def validate_content_type(
    content_type: str,
    allowed_modes: list[str],
) -> bool:
    """Validate that a content type is allowed by a list of modes.

    Args:
        content_type: The MIME type to validate.
        allowed_modes: List of allowed MIME types (may include wildcards).

    Returns:
        True if content_type is compatible with any allowed mode.
    """
    for mode in allowed_modes:
        if _mime_types_compatible(content_type, mode):
            return True
    return False


def get_part_content_type(part: Part) -> str:
    """Extract MIME type from an A2A Part.

    Args:
        part: A Part object containing TextPart, DataPart, or FilePart.

    Returns:
        The MIME type string for this part.
    """
    root = part.root
    if root.kind == "text":
        return TEXT_PLAIN
    if root.kind == "data":
        return APPLICATION_JSON
    if root.kind == "file":
        return root.file.mime_type or APPLICATION_OCTET_STREAM
    return APPLICATION_OCTET_STREAM


def validate_message_parts(
    parts: list[Part],
    allowed_modes: list[str],
) -> list[str]:
    """Validate that all message parts have allowed content types.

    Args:
        parts: List of Parts from the incoming message.
        allowed_modes: List of allowed MIME types (from default_input_modes).

    Returns:
        List of invalid content types found (empty if all valid).
    """
    invalid_types: list[str] = []
    for part in parts:
        content_type = get_part_content_type(part)
        if not validate_content_type(content_type, allowed_modes):
            if content_type not in invalid_types:
                invalid_types.append(content_type)
    return invalid_types
