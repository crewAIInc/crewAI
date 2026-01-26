"""Anthropic Files API uploader implementation."""

from __future__ import annotations

import logging
import os
from typing import Any

from crewai_files.core.sources import generate_filename
from crewai_files.core.types import FileInput
from crewai_files.processing.exceptions import classify_upload_error
from crewai_files.uploaders.base import FileUploader, UploadResult


logger = logging.getLogger(__name__)


class AnthropicFileUploader(FileUploader):
    """Uploader for Anthropic Files API.

    Uses the anthropic SDK to upload files. Files are stored persistently
    until explicitly deleted.
    """

    def __init__(
        self,
        api_key: str | None = None,
        client: Any = None,
        async_client: Any = None,
    ) -> None:
        """Initialize the Anthropic uploader.

        Args:
            api_key: Optional Anthropic API key. If not provided, uses
                ANTHROPIC_API_KEY environment variable.
            client: Optional pre-instantiated Anthropic client.
            async_client: Optional pre-instantiated async Anthropic client.
        """
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client: Any = client
        self._async_client: Any = async_client

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "anthropic"

    def _get_client(self) -> Any:
        """Get or create the Anthropic client."""
        if self._client is None:
            try:
                import anthropic

                self._client = anthropic.Anthropic(api_key=self._api_key)
            except ImportError as e:
                raise ImportError(
                    "anthropic is required for Anthropic file uploads. "
                    "Install with: pip install anthropic"
                ) from e
        return self._client

    def _get_async_client(self) -> Any:
        """Get or create the async Anthropic client."""
        if self._async_client is None:
            try:
                import anthropic

                self._async_client = anthropic.AsyncAnthropic(api_key=self._api_key)
            except ImportError as e:
                raise ImportError(
                    "anthropic is required for Anthropic file uploads. "
                    "Install with: pip install anthropic"
                ) from e
        return self._async_client

    def upload(self, file: FileInput, purpose: str | None = None) -> UploadResult:
        """Upload a file to Anthropic.

        Args:
            file: The file to upload.
            purpose: Optional purpose for the file (default: "user_upload").

        Returns:
            UploadResult with the file ID and metadata.

        Raises:
            TransientUploadError: For retryable errors (network, rate limits).
            PermanentUploadError: For non-retryable errors (auth, validation).
        """
        try:
            client = self._get_client()

            content = file.read()

            logger.info(
                f"Uploading file '{file.filename}' to Anthropic ({len(content)} bytes)"
            )

            filename = file.filename or generate_filename(file.content_type)
            uploaded_file = client.beta.files.upload(
                file=(filename, content, file.content_type),
            )

            logger.info(f"Uploaded to Anthropic: {uploaded_file.id}")

            return UploadResult(
                file_id=uploaded_file.id,
                file_uri=None,
                content_type=file.content_type,
                expires_at=None,
                provider=self.provider_name,
            )
        except ImportError:
            raise
        except Exception as e:
            raise classify_upload_error(e, file.filename) from e

    def delete(self, file_id: str) -> bool:
        """Delete an uploaded file from Anthropic.

        Args:
            file_id: The file ID to delete.

        Returns:
            True if deletion was successful, False otherwise.
        """
        try:
            client = self._get_client()
            client.beta.files.delete(file_id=file_id)
            logger.info(f"Deleted Anthropic file: {file_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete Anthropic file {file_id}: {e}")
            return False

    def get_file_info(self, file_id: str) -> dict[str, Any] | None:
        """Get information about an uploaded file.

        Args:
            file_id: The file ID.

        Returns:
            Dictionary with file information, or None if not found.
        """
        try:
            client = self._get_client()
            file_info = client.beta.files.retrieve(file_id=file_id)
            return {
                "id": file_info.id,
                "filename": file_info.filename,
                "purpose": file_info.purpose,
                "size_bytes": file_info.size_bytes,
                "created_at": file_info.created_at,
            }
        except Exception as e:
            logger.debug(f"Failed to get Anthropic file info for {file_id}: {e}")
            return None

    def list_files(self) -> list[dict[str, Any]]:
        """List all uploaded files.

        Returns:
            List of dictionaries with file information.
        """
        try:
            client = self._get_client()
            files = client.beta.files.list()
            return [
                {
                    "id": f.id,
                    "filename": f.filename,
                    "purpose": f.purpose,
                    "size_bytes": f.size_bytes,
                    "created_at": f.created_at,
                }
                for f in files.data
            ]
        except Exception as e:
            logger.warning(f"Failed to list Anthropic files: {e}")
            return []

    async def aupload(
        self, file: FileInput, purpose: str | None = None
    ) -> UploadResult:
        """Async upload a file to Anthropic using native async client.

        Args:
            file: The file to upload.
            purpose: Optional purpose for the file (default: "user_upload").

        Returns:
            UploadResult with the file ID and metadata.

        Raises:
            TransientUploadError: For retryable errors (network, rate limits).
            PermanentUploadError: For non-retryable errors (auth, validation).
        """
        try:
            client = self._get_async_client()

            content = await file.aread()

            logger.info(
                f"Uploading file '{file.filename}' to Anthropic ({len(content)} bytes)"
            )

            filename = file.filename or generate_filename(file.content_type)
            uploaded_file = await client.beta.files.upload(
                file=(filename, content, file.content_type),
            )

            logger.info(f"Uploaded to Anthropic: {uploaded_file.id}")

            return UploadResult(
                file_id=uploaded_file.id,
                file_uri=None,
                content_type=file.content_type,
                expires_at=None,
                provider=self.provider_name,
            )
        except ImportError:
            raise
        except Exception as e:
            raise classify_upload_error(e, file.filename) from e

    async def adelete(self, file_id: str) -> bool:
        """Async delete an uploaded file from Anthropic.

        Args:
            file_id: The file ID to delete.

        Returns:
            True if deletion was successful, False otherwise.
        """
        try:
            client = self._get_async_client()
            await client.beta.files.delete(file_id=file_id)
            logger.info(f"Deleted Anthropic file: {file_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete Anthropic file {file_id}: {e}")
            return False
