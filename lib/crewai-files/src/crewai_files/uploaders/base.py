"""Base class for file uploaders."""

from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from crewai_files.core.types import FileInput


@dataclass
class UploadResult:
    """Result of a file upload operation.

    Attributes:
        file_id: Provider-specific file identifier.
        file_uri: Optional URI for accessing the file.
        content_type: MIME type of the uploaded file.
        expires_at: When the upload expires (if applicable).
        provider: Name of the provider.
    """

    file_id: str
    provider: str
    content_type: str
    file_uri: str | None = None
    expires_at: datetime | None = None


class FileUploader(ABC):
    """Abstract base class for provider file uploaders.

    Implementations handle uploading files to provider-specific File APIs.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""

    @abstractmethod
    def upload(self, file: FileInput, purpose: str | None = None) -> UploadResult:
        """Upload a file to the provider.

        Args:
            file: The file to upload.
            purpose: Optional purpose/description for the upload.

        Returns:
            UploadResult with the file identifier and metadata.

        Raises:
            Exception: If upload fails.
        """

    async def aupload(
        self, file: FileInput, purpose: str | None = None
    ) -> UploadResult:
        """Async upload a file to the provider.

        Default implementation runs sync upload in executor.
        Override in subclasses for native async support.

        Args:
            file: The file to upload.
            purpose: Optional purpose/description for the upload.

        Returns:
            UploadResult with the file identifier and metadata.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.upload, file, purpose)

    @abstractmethod
    def delete(self, file_id: str) -> bool:
        """Delete an uploaded file.

        Args:
            file_id: The file identifier to delete.

        Returns:
            True if deletion was successful, False otherwise.
        """

    async def adelete(self, file_id: str) -> bool:
        """Async delete an uploaded file.

        Default implementation runs sync delete in executor.
        Override in subclasses for native async support.

        Args:
            file_id: The file identifier to delete.

        Returns:
            True if deletion was successful, False otherwise.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.delete, file_id)

    def get_file_info(self, file_id: str) -> dict[str, Any] | None:
        """Get information about an uploaded file.

        Args:
            file_id: The file identifier.

        Returns:
            Dictionary with file information, or None if not found.
        """
        return None

    def list_files(self) -> list[dict[str, Any]]:
        """List all uploaded files.

        Returns:
            List of dictionaries with file information.
        """
        return []
