"""Gemini File API uploader implementation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import io
import logging
import os
from typing import Any

from crewai.utilities.files.content_types import (
    AudioFile,
    ImageFile,
    PDFFile,
    TextFile,
    VideoFile,
)
from crewai.utilities.files.uploaders.base import FileUploader, UploadResult


logger = logging.getLogger(__name__)

FileInput = AudioFile | ImageFile | PDFFile | TextFile | VideoFile

# Gemini files expire after 48 hours
GEMINI_FILE_TTL = timedelta(hours=48)


class GeminiFileUploader(FileUploader):
    """Uploader for Google Gemini File API.

    Uses the google-genai SDK to upload files. Files are stored for 48 hours.

    Attributes:
        api_key: Optional API key (uses GOOGLE_API_KEY env var if not provided).
    """

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize the Gemini uploader.

        Args:
            api_key: Optional Google API key. If not provided, uses
                GOOGLE_API_KEY environment variable.
        """
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self._client: Any = None

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "gemini"

    def _get_client(self) -> Any:
        """Get or create the Gemini client."""
        if self._client is None:
            try:
                from google import genai

                self._client = genai.Client(api_key=self._api_key)
            except ImportError as e:
                raise ImportError(
                    "google-genai is required for Gemini file uploads. "
                    "Install with: pip install google-genai"
                ) from e
        return self._client

    def upload(self, file: FileInput, purpose: str | None = None) -> UploadResult:
        """Upload a file to Gemini.

        Args:
            file: The file to upload.
            purpose: Optional purpose/description (used as display name).

        Returns:
            UploadResult with the file URI and metadata.

        Raises:
            Exception: If upload fails.
        """
        client = self._get_client()

        content = file.source.read()
        display_name = purpose or file.filename

        # Create a file-like object for upload
        file_data = io.BytesIO(content)
        file_data.name = file.filename

        logger.info(
            f"Uploading file '{file.filename}' to Gemini ({len(content)} bytes)"
        )

        # Upload using the genai client
        uploaded_file = client.files.upload(
            file=file_data,
            config={
                "display_name": display_name,
                "mime_type": file.content_type,
            },
        )

        expires_at = datetime.now(timezone.utc) + GEMINI_FILE_TTL

        logger.info(
            f"Uploaded to Gemini: {uploaded_file.name} (URI: {uploaded_file.uri})"
        )

        return UploadResult(
            file_id=uploaded_file.name,
            file_uri=uploaded_file.uri,
            content_type=file.content_type,
            expires_at=expires_at,
            provider=self.provider_name,
        )

    def delete(self, file_id: str) -> bool:
        """Delete an uploaded file from Gemini.

        Args:
            file_id: The file name/ID to delete.

        Returns:
            True if deletion was successful, False otherwise.
        """
        try:
            client = self._get_client()
            client.files.delete(name=file_id)
            logger.info(f"Deleted Gemini file: {file_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete Gemini file {file_id}: {e}")
            return False

    def get_file_info(self, file_id: str) -> dict[str, Any] | None:
        """Get information about an uploaded file.

        Args:
            file_id: The file name/ID.

        Returns:
            Dictionary with file information, or None if not found.
        """
        try:
            client = self._get_client()
            file_info = client.files.get(name=file_id)
            return {
                "name": file_info.name,
                "uri": file_info.uri,
                "display_name": file_info.display_name,
                "mime_type": file_info.mime_type,
                "size_bytes": file_info.size_bytes,
                "state": str(file_info.state),
                "create_time": file_info.create_time,
                "expiration_time": file_info.expiration_time,
            }
        except Exception as e:
            logger.debug(f"Failed to get Gemini file info for {file_id}: {e}")
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
                    "name": f.name,
                    "uri": f.uri,
                    "display_name": f.display_name,
                    "mime_type": f.mime_type,
                    "size_bytes": f.size_bytes,
                    "state": str(f.state),
                }
                for f in files
            ]
        except Exception as e:
            logger.warning(f"Failed to list Gemini files: {e}")
            return []

    def wait_for_processing(self, file_id: str, timeout_seconds: int = 300) -> bool:
        """Wait for a file to finish processing.

        Some files (especially videos) need time to process after upload.

        Args:
            file_id: The file name/ID.
            timeout_seconds: Maximum time to wait.

        Returns:
            True if processing completed, False if timed out or failed.
        """
        import time

        try:
            from google.genai.types import FileState
        except ImportError:
            # If we can't import FileState, just return True
            return True

        client = self._get_client()
        start_time = time.time()

        while time.time() - start_time < timeout_seconds:
            file_info = client.files.get(name=file_id)

            if file_info.state == FileState.ACTIVE:
                return True

            if file_info.state == FileState.FAILED:
                logger.error(f"Gemini file processing failed: {file_id}")
                return False

            time.sleep(2)

        logger.warning(f"Timed out waiting for Gemini file processing: {file_id}")
        return False
