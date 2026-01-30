"""Resolved file types representing different delivery methods for file content."""

from abc import ABC
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ResolvedFile(ABC):
    """Base class for resolved file representations.

    A ResolvedFile represents the final form of a file ready for delivery
    to an LLM provider, whether inline or via reference.

    Attributes:
        content_type: MIME type of the file content.
    """

    content_type: str


@dataclass(frozen=True)
class InlineBase64(ResolvedFile):
    """File content encoded as base64 string.

    Used by most providers for inline file content in messages.

    Attributes:
        content_type: MIME type of the file content.
        data: Base64-encoded file content.
    """

    data: str


@dataclass(frozen=True)
class InlineBytes(ResolvedFile):
    """File content as raw bytes.

    Used by providers like Bedrock that accept raw bytes instead of base64.

    Attributes:
        content_type: MIME type of the file content.
        data: Raw file bytes.
    """

    data: bytes


@dataclass(frozen=True)
class FileReference(ResolvedFile):
    """Reference to an uploaded file.

    Used when files are uploaded via provider File APIs.

    Attributes:
        content_type: MIME type of the file content.
        file_id: Provider-specific file identifier.
        provider: Name of the provider the file was uploaded to.
        expires_at: When the uploaded file expires (if applicable).
        file_uri: Optional URI for accessing the file (used by Gemini).
    """

    file_id: str
    provider: str
    expires_at: datetime | None = None
    file_uri: str | None = None


@dataclass(frozen=True)
class UrlReference(ResolvedFile):
    """Reference to a file accessible via URL.

    Used by providers that support fetching files from URLs.

    Attributes:
        content_type: MIME type of the file content.
        url: URL where the file can be accessed.
    """

    url: str


ResolvedFileType = InlineBase64 | InlineBytes | FileReference | UrlReference
