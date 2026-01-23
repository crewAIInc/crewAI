"""OpenAI Files API uploader implementation."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
import io
import logging
import os
from typing import Any

from crewai_files.core.constants import DEFAULT_UPLOAD_CHUNK_SIZE, FILES_API_MAX_SIZE
from crewai_files.core.sources import FileBytes, FilePath, FileStream, generate_filename
from crewai_files.core.types import FileInput
from crewai_files.processing.exceptions import (
    PermanentUploadError,
    TransientUploadError,
    classify_upload_error,
)
from crewai_files.uploaders.base import FileUploader, UploadResult


logger = logging.getLogger(__name__)


def _get_purpose_for_content_type(content_type: str, purpose: str | None) -> str:
    """Get the appropriate purpose for a file based on content type.

    OpenAI Files API requires different purposes for different file types:
    - Images (for Responses API vision): "vision"
    - PDFs and other documents: "user_data"

    Args:
        content_type: MIME type of the file.
        purpose: Optional explicit purpose override.

    Returns:
        The purpose string to use for upload.
    """
    if purpose is not None:
        return purpose
    if content_type.startswith("image/"):
        return "vision"
    return "user_data"


def _get_file_size(file: FileInput) -> int | None:
    """Get file size without reading content if possible.

    Args:
        file: The file to get size for.

    Returns:
        File size in bytes, or None if size cannot be determined without reading.
    """
    source = file._file_source
    if isinstance(source, FilePath):
        return source.path.stat().st_size
    if isinstance(source, FileBytes):
        return len(source.data)
    return None


def _iter_file_chunks(file: FileInput, chunk_size: int) -> Iterator[bytes]:
    """Iterate over file content in chunks.

    Args:
        file: The file to read.
        chunk_size: Size of each chunk in bytes.

    Yields:
        Chunks of file content.
    """
    source = file._file_source
    if isinstance(source, (FilePath, FileBytes, FileStream)):
        yield from source.read_chunks(chunk_size)
    else:
        content = file.read()
        for i in range(0, len(content), chunk_size):
            yield content[i : i + chunk_size]


async def _aiter_file_chunks(
    file: FileInput, chunk_size: int, content: bytes | None = None
) -> AsyncIterator[bytes]:
    """Async iterate over file content in chunks.

    Args:
        file: The file to read.
        chunk_size: Size of each chunk in bytes.
        content: Optional pre-loaded content to chunk.

    Yields:
        Chunks of file content.
    """
    if content is not None:
        for i in range(0, len(content), chunk_size):
            yield content[i : i + chunk_size]
        return

    source = file._file_source
    if isinstance(source, FilePath):
        async for chunk in source.aread_chunks(chunk_size):
            yield chunk
    elif isinstance(source, (FileBytes, FileStream)):
        for chunk in source.read_chunks(chunk_size):
            yield chunk
    else:
        data = await file.aread()
        for i in range(0, len(data), chunk_size):
            yield data[i : i + chunk_size]


class OpenAIFileUploader(FileUploader):
    """Uploader for OpenAI Files and Uploads APIs.

    Uses the Files API for files up to 512MB (single request).
    Uses the Uploads API for files larger than 512MB (multipart chunked).
    """

    def __init__(
        self,
        api_key: str | None = None,
        chunk_size: int = DEFAULT_UPLOAD_CHUNK_SIZE,
        client: Any = None,
        async_client: Any = None,
    ) -> None:
        """Initialize the OpenAI uploader.

        Args:
            api_key: Optional OpenAI API key. If not provided, uses
                OPENAI_API_KEY environment variable.
            chunk_size: Chunk size in bytes for multipart uploads (default 64MB).
            client: Optional pre-instantiated OpenAI client.
            async_client: Optional pre-instantiated async OpenAI client.
        """
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._chunk_size = chunk_size
        self._client: Any = client
        self._async_client: Any = async_client

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "openai"

    def _build_upload_result(self, file_id: str, content_type: str) -> UploadResult:
        """Build an UploadResult for a completed upload.

        Args:
            file_id: The uploaded file ID.
            content_type: The file's content type.

        Returns:
            UploadResult with the file metadata.
        """
        return UploadResult(
            file_id=file_id,
            file_uri=None,
            content_type=content_type,
            expires_at=None,
            provider=self.provider_name,
        )

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

    def _get_async_client(self) -> Any:
        """Get or create the async OpenAI client."""
        if self._async_client is None:
            try:
                from openai import AsyncOpenAI

                self._async_client = AsyncOpenAI(api_key=self._api_key)
            except ImportError as e:
                raise ImportError(
                    "openai is required for OpenAI file uploads. "
                    "Install with: pip install openai"
                ) from e
        return self._async_client

    def upload(self, file: FileInput, purpose: str | None = None) -> UploadResult:
        """Upload a file to OpenAI.

        Uses Files API for files <= 512MB, Uploads API for larger files.
        For large files, streams chunks to avoid loading entire file in memory.

        Args:
            file: The file to upload.
            purpose: Optional purpose for the file (default: "user_data").

        Returns:
            UploadResult with the file ID and metadata.

        Raises:
            TransientUploadError: For retryable errors (network, rate limits).
            PermanentUploadError: For non-retryable errors (auth, validation).
        """
        try:
            file_size = _get_file_size(file)

            if file_size is not None and file_size > FILES_API_MAX_SIZE:
                return self._upload_multipart_streaming(file, file_size, purpose)

            content = file.read()
            if len(content) > FILES_API_MAX_SIZE:
                return self._upload_multipart(file, content, purpose)
            return self._upload_simple(file, content, purpose)
        except ImportError:
            raise
        except (TransientUploadError, PermanentUploadError):
            raise
        except Exception as e:
            raise classify_upload_error(e, file.filename) from e

    def _upload_simple(
        self,
        file: FileInput,
        content: bytes,
        purpose: str | None,
    ) -> UploadResult:
        """Upload using the Files API (single request, up to 512MB).

        Args:
            file: The file to upload.
            content: File content bytes.
            purpose: Optional purpose for the file.

        Returns:
            UploadResult with the file ID and metadata.
        """
        client = self._get_client()
        file_purpose = _get_purpose_for_content_type(file.content_type, purpose)
        filename = file.filename or generate_filename(file.content_type)

        file_data = io.BytesIO(content)
        file_data.name = filename

        logger.info(
            f"Uploading file '{filename}' to OpenAI Files API ({len(content)} bytes)"
        )

        uploaded_file = client.files.create(
            file=file_data,
            purpose=file_purpose,
        )

        logger.info(f"Uploaded to OpenAI: {uploaded_file.id}")

        return self._build_upload_result(uploaded_file.id, file.content_type)

    def _upload_multipart(
        self,
        file: FileInput,
        content: bytes,
        purpose: str | None,
    ) -> UploadResult:
        """Upload using the Uploads API with content already in memory.

        Args:
            file: The file to upload.
            content: File content bytes (already loaded).
            purpose: Optional purpose for the file.

        Returns:
            UploadResult with the file ID and metadata.
        """
        client = self._get_client()
        file_purpose = _get_purpose_for_content_type(file.content_type, purpose)
        filename = file.filename or generate_filename(file.content_type)
        file_size = len(content)

        logger.info(
            f"Uploading file '{filename}' to OpenAI Uploads API "
            f"({file_size} bytes, {self._chunk_size} byte chunks)"
        )

        upload = client.uploads.create(
            bytes=file_size,
            filename=filename,
            mime_type=file.content_type,
            purpose=file_purpose,
        )

        part_ids: list[str] = []
        offset = 0
        part_num = 1

        try:
            while offset < file_size:
                chunk = content[offset : offset + self._chunk_size]
                chunk_io = io.BytesIO(chunk)

                logger.debug(
                    f"Uploading part {part_num} ({len(chunk)} bytes, offset {offset})"
                )

                part = client.uploads.parts.create(
                    upload_id=upload.id,
                    data=chunk_io,
                )
                part_ids.append(part.id)

                offset += self._chunk_size
                part_num += 1

            completed = client.uploads.complete(
                upload_id=upload.id,
                part_ids=part_ids,
            )

            file_id = completed.file.id if completed.file else upload.id
            logger.info(f"Completed multipart upload to OpenAI: {file_id}")

            return self._build_upload_result(file_id, file.content_type)
        except Exception:
            logger.warning(f"Multipart upload failed, cancelling upload {upload.id}")
            try:
                client.uploads.cancel(upload_id=upload.id)
            except Exception as cancel_err:
                logger.debug(f"Failed to cancel upload: {cancel_err}")
            raise

    def _upload_multipart_streaming(
        self,
        file: FileInput,
        file_size: int,
        purpose: str | None,
    ) -> UploadResult:
        """Upload using the Uploads API with streaming chunks.

        Streams chunks directly from the file source without loading
        the entire file into memory. Used for large files.

        Args:
            file: The file to upload.
            file_size: Total file size in bytes.
            purpose: Optional purpose for the file.

        Returns:
            UploadResult with the file ID and metadata.
        """
        client = self._get_client()
        file_purpose = _get_purpose_for_content_type(file.content_type, purpose)
        filename = file.filename or generate_filename(file.content_type)

        logger.info(
            f"Uploading file '{filename}' to OpenAI Uploads API (streaming) "
            f"({file_size} bytes, {self._chunk_size} byte chunks)"
        )

        upload = client.uploads.create(
            bytes=file_size,
            filename=filename,
            mime_type=file.content_type,
            purpose=file_purpose,
        )

        part_ids: list[str] = []
        part_num = 1

        try:
            for chunk in _iter_file_chunks(file, self._chunk_size):
                chunk_io = io.BytesIO(chunk)

                logger.debug(f"Uploading part {part_num} ({len(chunk)} bytes)")

                part = client.uploads.parts.create(
                    upload_id=upload.id,
                    data=chunk_io,
                )
                part_ids.append(part.id)
                part_num += 1

            completed = client.uploads.complete(
                upload_id=upload.id,
                part_ids=part_ids,
            )

            file_id = completed.file.id if completed.file else upload.id
            logger.info(f"Completed streaming multipart upload to OpenAI: {file_id}")

            return self._build_upload_result(file_id, file.content_type)
        except Exception:
            logger.warning(f"Multipart upload failed, cancelling upload {upload.id}")
            try:
                client.uploads.cancel(upload_id=upload.id)
            except Exception as cancel_err:
                logger.debug(f"Failed to cancel upload: {cancel_err}")
            raise

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

    async def aupload(
        self, file: FileInput, purpose: str | None = None
    ) -> UploadResult:
        """Async upload a file to OpenAI using native async client.

        Uses Files API for files <= 512MB, Uploads API for larger files.
        For large files, streams chunks to avoid loading entire file in memory.

        Args:
            file: The file to upload.
            purpose: Optional purpose for the file (default: "user_data").

        Returns:
            UploadResult with the file ID and metadata.

        Raises:
            TransientUploadError: For retryable errors (network, rate limits).
            PermanentUploadError: For non-retryable errors (auth, validation).
        """
        try:
            file_size = _get_file_size(file)

            if file_size is not None and file_size > FILES_API_MAX_SIZE:
                return await self._aupload_multipart_streaming(file, file_size, purpose)

            content = await file.aread()
            if len(content) > FILES_API_MAX_SIZE:
                return await self._aupload_multipart(file, content, purpose)
            return await self._aupload_simple(file, content, purpose)
        except ImportError:
            raise
        except (TransientUploadError, PermanentUploadError):
            raise
        except Exception as e:
            raise classify_upload_error(e, file.filename) from e

    async def _aupload_simple(
        self,
        file: FileInput,
        content: bytes,
        purpose: str | None,
    ) -> UploadResult:
        """Async upload using the Files API (single request, up to 512MB).

        Args:
            file: The file to upload.
            content: File content bytes.
            purpose: Optional purpose for the file.

        Returns:
            UploadResult with the file ID and metadata.
        """
        client = self._get_async_client()
        file_purpose = _get_purpose_for_content_type(file.content_type, purpose)

        file_data = io.BytesIO(content)
        file_data.name = file.filename or generate_filename(file.content_type)

        logger.info(
            f"Uploading file '{file.filename}' to OpenAI Files API ({len(content)} bytes)"
        )

        uploaded_file = await client.files.create(
            file=file_data,
            purpose=file_purpose,
        )

        logger.info(f"Uploaded to OpenAI: {uploaded_file.id}")

        return self._build_upload_result(uploaded_file.id, file.content_type)

    async def _aupload_multipart(
        self,
        file: FileInput,
        content: bytes,
        purpose: str | None,
    ) -> UploadResult:
        """Async upload using the Uploads API (multipart chunked, up to 8GB).

        Args:
            file: The file to upload.
            content: File content bytes.
            purpose: Optional purpose for the file.

        Returns:
            UploadResult with the file ID and metadata.
        """
        client = self._get_async_client()
        file_purpose = _get_purpose_for_content_type(file.content_type, purpose)
        filename = file.filename or generate_filename(file.content_type)
        file_size = len(content)

        logger.info(
            f"Uploading file '{filename}' to OpenAI Uploads API "
            f"({file_size} bytes, {self._chunk_size} byte chunks)"
        )

        upload = await client.uploads.create(
            bytes=file_size,
            filename=filename,
            mime_type=file.content_type,
            purpose=file_purpose,
        )

        part_ids: list[str] = []
        offset = 0
        part_num = 1

        try:
            while offset < file_size:
                chunk = content[offset : offset + self._chunk_size]
                chunk_io = io.BytesIO(chunk)

                logger.debug(
                    f"Uploading part {part_num} ({len(chunk)} bytes, offset {offset})"
                )

                part = await client.uploads.parts.create(
                    upload_id=upload.id,
                    data=chunk_io,
                )
                part_ids.append(part.id)

                offset += self._chunk_size
                part_num += 1

            completed = await client.uploads.complete(
                upload_id=upload.id,
                part_ids=part_ids,
            )

            file_id = completed.file.id if completed.file else upload.id
            logger.info(f"Completed multipart upload to OpenAI: {file_id}")

            return self._build_upload_result(file_id, file.content_type)
        except Exception:
            logger.warning(f"Multipart upload failed, cancelling upload {upload.id}")
            try:
                await client.uploads.cancel(upload_id=upload.id)
            except Exception as cancel_err:
                logger.debug(f"Failed to cancel upload: {cancel_err}")
            raise

    async def _aupload_multipart_streaming(
        self,
        file: FileInput,
        file_size: int,
        purpose: str | None,
    ) -> UploadResult:
        """Async upload using the Uploads API with streaming chunks.

        Streams chunks directly from the file source without loading
        the entire file into memory. Used for large files.

        Args:
            file: The file to upload.
            file_size: Total file size in bytes.
            purpose: Optional purpose for the file.

        Returns:
            UploadResult with the file ID and metadata.
        """
        client = self._get_async_client()
        file_purpose = _get_purpose_for_content_type(file.content_type, purpose)
        filename = file.filename or generate_filename(file.content_type)

        logger.info(
            f"Uploading file '{filename}' to OpenAI Uploads API (streaming) "
            f"({file_size} bytes, {self._chunk_size} byte chunks)"
        )

        upload = await client.uploads.create(
            bytes=file_size,
            filename=filename,
            mime_type=file.content_type,
            purpose=file_purpose,
        )

        part_ids: list[str] = []
        part_num = 1

        try:
            async for chunk in _aiter_file_chunks(file, self._chunk_size):
                chunk_io = io.BytesIO(chunk)

                logger.debug(f"Uploading part {part_num} ({len(chunk)} bytes)")

                part = await client.uploads.parts.create(
                    upload_id=upload.id,
                    data=chunk_io,
                )
                part_ids.append(part.id)
                part_num += 1

            completed = await client.uploads.complete(
                upload_id=upload.id,
                part_ids=part_ids,
            )

            file_id = completed.file.id if completed.file else upload.id
            logger.info(f"Completed streaming multipart upload to OpenAI: {file_id}")

            return self._build_upload_result(file_id, file.content_type)
        except Exception:
            logger.warning(f"Multipart upload failed, cancelling upload {upload.id}")
            try:
                await client.uploads.cancel(upload_id=upload.id)
            except Exception as cancel_err:
                logger.debug(f"Failed to cancel upload: {cancel_err}")
            raise

    async def adelete(self, file_id: str) -> bool:
        """Async delete an uploaded file from OpenAI.

        Args:
            file_id: The file ID to delete.

        Returns:
            True if deletion was successful, False otherwise.
        """
        try:
            client = self._get_async_client()
            await client.files.delete(file_id)
            logger.info(f"Deleted OpenAI file: {file_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete OpenAI file {file_id}: {e}")
            return False
