"""Type definitions for A2A protocol message parts."""

from __future__ import annotations

from typing import (
    Annotated,
    Any,
    Literal,
    Protocol,
    TypedDict,
    runtime_checkable,
)

from pydantic import BeforeValidator, HttpUrl, TypeAdapter
from typing_extensions import NotRequired


try:
    from crewai.a2a.updates import (
        PollingConfig,
        PollingHandler,
        PushNotificationConfig,
        PushNotificationHandler,
        StreamingConfig,
        StreamingHandler,
        UpdateConfig,
    )
except ImportError:
    PollingConfig = Any  # type: ignore[misc,assignment]
    PollingHandler = Any  # type: ignore[misc,assignment]
    PushNotificationConfig = Any  # type: ignore[misc,assignment]
    PushNotificationHandler = Any  # type: ignore[misc,assignment]
    StreamingConfig = Any  # type: ignore[misc,assignment]
    StreamingHandler = Any  # type: ignore[misc,assignment]
    UpdateConfig = Any  # type: ignore[misc,assignment]


TransportType = Literal["JSONRPC", "GRPC", "HTTP+JSON"]
ProtocolVersion = Literal[
    "0.2.0",
    "0.2.1",
    "0.2.2",
    "0.2.3",
    "0.2.4",
    "0.2.5",
    "0.2.6",
    "0.3.0",
    "0.4.0",
]

http_url_adapter: TypeAdapter[HttpUrl] = TypeAdapter(HttpUrl)

Url = Annotated[
    str,
    BeforeValidator(
        lambda value: str(http_url_adapter.validate_python(value, strict=True))
    ),
]


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


PollingHandlerType = type[PollingHandler]
StreamingHandlerType = type[StreamingHandler]
PushNotificationHandlerType = type[PushNotificationHandler]

HandlerType = PollingHandlerType | StreamingHandlerType | PushNotificationHandlerType

HANDLER_REGISTRY: dict[type[UpdateConfig], HandlerType] = {
    PollingConfig: PollingHandler,
    StreamingConfig: StreamingHandler,
    PushNotificationConfig: PushNotificationHandler,
}
