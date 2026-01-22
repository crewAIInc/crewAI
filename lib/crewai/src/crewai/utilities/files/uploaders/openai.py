"""OpenAI Files API uploader implementation."""

from __future__ import annotations

import io
import logging
import os
from typing import Any

from crewai.utilities.files.content_types import (
    AudioFile,
    File,
    ImageFile,
    PDFFile,
    TextFile,
    VideoFile,
)
from crewai.utilities.files.uploaders.base import FileUploader, UploadResult


logger = logging.getLogger(__name__)

FileInput = AudioFile | File | ImageFile | PDFFile | TextFile | VideoFile


class OpenAIFileUploader(FileUploader):
    """Uploader for OpenAI Files API.

    Uses the OpenAI SDK to upload files. Files are stored persistently
    until explicitly deleted.

    Attributes:
        api_key: Optional API key (uses OPENAI_API_KEY env var if not provided).
    """

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize the OpenAI uploader.

        Args:
            api_key: Optional OpenAI API key. If not provided, uses
                OPENAI_API_KEY environment variable.
        """
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client: Any = None

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "openai"

    def _get_client(self) -> Any:
        """Get or create the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=self._api_key)
            except ImportError as e:
                raise ImportError(
                    "openai is required for OpenAI file uploads. "
                    "Install with: pip install openai"
                ) from e
        return self._client

    def upload(self, file: FileInput, purpose: str | None = None) -> UploadResult:
        """Upload a file to OpenAI.

        Args:
            file: The file to upload.
            purpose: Optional purpose for the file (default: "user_data").

        Returns:
            UploadResult with the file ID and metadata.

        Raises:
            Exception: If upload fails.
        """
        client = self._get_client()

        content = file.read()
        file_purpose = purpose or "user_data"

        file_data = io.BytesIO(content)
        file_data.name = file.filename or "file"

        logger.info(
            f"Uploading file '{file.filename}' to OpenAI ({len(content)} bytes)"
        )

        uploaded_file = client.files.create(
            file=file_data,
            purpose=file_purpose,
        )

        logger.info(f"Uploaded to OpenAI: {uploaded_file.id}")

        return UploadResult(
            file_id=uploaded_file.id,
            file_uri=None,
            content_type=file.content_type,
            expires_at=None,
            provider=self.provider_name,
        )

    def delete(self, file_id: str) -> bool:
        """Delete an uploaded file from OpenAI.

        Args:
            file_id: The file ID to delete.

        Returns:
            True if deletion was successful, False otherwise.
        """
        try:
            client = self._get_client()
            client.files.delete(file_id)
            logger.info(f"Deleted OpenAI file: {file_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete OpenAI file {file_id}: {e}")
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
            file_info = client.files.retrieve(file_id)
            return {
                "id": file_info.id,
                "filename": file_info.filename,
                "purpose": file_info.purpose,
                "bytes": file_info.bytes,
                "created_at": file_info.created_at,
                "status": file_info.status,
            }
        except Exception as e:
            logger.debug(f"Failed to get OpenAI file info for {file_id}: {e}")
            return None

    def list_files(self) -> list[dict[str, Any]]:
        """List all uploaded files.

        Returns:
            List of dictionaries with file information.
        """
        try:
            client = self._get_client()
            files = client.files.list()
            return [
                {
                    "id": f.id,
                    "filename": f.filename,
                    "purpose": f.purpose,
                    "bytes": f.bytes,
                    "created_at": f.created_at,
                    "status": f.status,
                }
                for f in files.data
            ]
        except Exception as e:
            logger.warning(f"Failed to list OpenAI files: {e}")
            return []
