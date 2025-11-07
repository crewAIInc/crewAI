"""Type definitions for A2A protocol message parts."""

from typing import Any, Literal, Protocol, TypedDict, runtime_checkable

from typing_extensions import NotRequired


@runtime_checkable
class AgentResponseProtocol(Protocol):
    """Protocol for the dynamically created AgentResponse model."""

    a2a_ids: tuple[str, ...]
    message: str
    is_a2a: bool


class PartsMetadataDict(TypedDict, total=False):
    """Metadata for A2A message parts.

    Attributes:
        mimeType: MIME type for the part content.
        schema: JSON schema for the part content.
    """

    mimeType: Literal["application/json"]
    schema: dict[str, Any]


class PartsDict(TypedDict):
    """A2A message part containing text and optional metadata.

    Attributes:
        text: The text content of the message part.
        metadata: Optional metadata describing the part content.
    """

    text: str
    metadata: NotRequired[PartsMetadataDict]
