"""AWS Bedrock S3 file uploader implementation."""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any

from crewai_files.core.constants import (
    MAX_CONCURRENCY,
    MULTIPART_CHUNKSIZE,
    MULTIPART_THRESHOLD,
)
from crewai_files.core.sources import FileBytes, FilePath
from crewai_files.core.types import FileInput
from crewai_files.processing.exceptions import (
    PermanentUploadError,
    TransientUploadError,
)
from crewai_files.uploaders.base import FileUploader, UploadResult


logger = logging.getLogger(__name__)


def _classify_s3_error(e: Exception, filename: str | None) -> Exception:
    """Classify an S3 exception as transient or permanent upload error.

    Args:
        e: The exception to classify.
        filename: The filename for error context.

    Returns:
        A TransientUploadError or PermanentUploadError wrapping the original.
    """
    error_type = type(e).__name__
    error_code = getattr(e, "response", {}).get("Error", {}).get("Code", "")

    if error_code in ("SlowDown", "ServiceUnavailable", "InternalError"):
        return TransientUploadError(f"Transient S3 error: {e}", file_name=filename)
    if error_code in ("AccessDenied", "InvalidAccessKeyId", "SignatureDoesNotMatch"):
        return PermanentUploadError(f"S3 authentication error: {e}", file_name=filename)
    if error_code in ("NoSuchBucket", "InvalidBucketName"):
        return PermanentUploadError(f"S3 bucket error: {e}", file_name=filename)
    if "Throttl" in error_type or "Throttl" in str(e):
        return TransientUploadError(f"S3 throttling: {e}", file_name=filename)
    return TransientUploadError(f"S3 upload failed: {e}", file_name=filename)


def _get_file_path(file: FileInput) -> Path | None:
    """Get the filesystem path if file source is FilePath.

    Args:
        file: The file input to check.

    Returns:
        Path if source is FilePath, None otherwise.
    """
    source = file._file_source
    if isinstance(source, FilePath):
        return source.path
    return None


def _get_file_size(file: FileInput) -> int | None:
    """Get file size without reading content if possible.

    Args:
        file: The file input.

    Returns:
        Size in bytes if determinable without reading, None otherwise.
    """
    source = file._file_source
    if isinstance(source, FilePath):
        return source.path.stat().st_size
    if isinstance(source, FileBytes):
        return len(source.data)
    return None


def _compute_hash_streaming(file_path: Path) -> str:
    """Compute SHA-256 hash by streaming file content.

    Args:
        file_path: Path to the file.

    Returns:
        First 16 characters of hex digest.
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


class BedrockFileUploader(FileUploader):
    """Uploader for AWS Bedrock via S3.

    Uploads files to S3 and returns S3 URIs that can be used with Bedrock's
    Converse API s3Location source format.
    """

    def __init__(
        self,
        bucket_name: str | None = None,
        bucket_owner: str | None = None,
        prefix: str = "crewai-files",
        region: str | None = None,
        client: Any = None,
        async_client: Any = None,
    ) -> None:
        """Initialize the Bedrock S3 uploader.

        Args:
            bucket_name: S3 bucket name. If not provided, uses
                CREWAI_BEDROCK_S3_BUCKET environment variable.
            bucket_owner: Optional bucket owner account ID for cross-account access.
                Uses CREWAI_BEDROCK_S3_BUCKET_OWNER environment variable if not provided.
            prefix: S3 key prefix for uploaded files (default: "crewai-files").
            region: AWS region. Uses AWS_REGION or AWS_DEFAULT_REGION if not provided.
            client: Optional pre-instantiated boto3 S3 client.
            async_client: Optional pre-instantiated aioboto3 S3 client.
        """
        self._bucket_name = bucket_name or os.environ.get("CREWAI_BEDROCK_S3_BUCKET")
        self._bucket_owner = bucket_owner or os.environ.get(
            "CREWAI_BEDROCK_S3_BUCKET_OWNER"
        )
        self._prefix = prefix
        self._region = region or os.environ.get(
            "AWS_REGION", os.environ.get("AWS_DEFAULT_REGION")
        )
        self._client: Any = client
        self._async_client: Any = async_client

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "bedrock"

    @property
    def bucket_name(self) -> str:
        """Return the configured bucket name."""
        if not self._bucket_name:
            raise ValueError(
                "S3 bucket name not configured. Set CREWAI_BEDROCK_S3_BUCKET "
                "environment variable or pass bucket_name parameter."
            )
        return self._bucket_name

    @property
    def bucket_owner(self) -> str | None:
        """Return the configured bucket owner."""
        return self._bucket_owner

    def _get_client(self) -> Any:
        """Get or create the S3 client."""
        if self._client is None:
            try:
                import boto3

                self._client = boto3.client("s3", region_name=self._region)
            except ImportError as e:
                raise ImportError(
                    "boto3 is required for Bedrock S3 file uploads. "
                    "Install with: pip install boto3"
                ) from e
        return self._client

    def _get_async_client(self) -> Any:
        """Get or create the async S3 client."""
        if self._async_client is None:
            try:
                import aioboto3  # type: ignore[import-not-found]

                self._session = aioboto3.Session()
            except ImportError as e:
                raise ImportError(
                    "aioboto3 is required for async Bedrock S3 file uploads. "
                    "Install with: pip install aioboto3"
                ) from e
        return self._session

    def _generate_s3_key(self, file: FileInput, content: bytes | None = None) -> str:
        """Generate a unique S3 key for the file.

        For FilePath sources with no content provided, computes hash via streaming.

        Args:
            file: The file being uploaded.
            content: The file content bytes (optional for FilePath sources).

        Returns:
            S3 key string.
        """
        if content is not None:
            content_hash = hashlib.sha256(content).hexdigest()[:16]
        else:
            file_path = _get_file_path(file)
            if file_path is not None:
                content_hash = _compute_hash_streaming(file_path)
            else:
                content_hash = hashlib.sha256(file.read()).hexdigest()[:16]

        filename = file.filename or "file"
        safe_filename = "".join(
            c if c.isalnum() or c in ".-_" else "_" for c in filename
        )
        return f"{self._prefix}/{content_hash}_{safe_filename}"

    def _build_s3_uri(self, key: str) -> str:
        """Build an S3 URI from a key.

        Args:
            key: The S3 object key.

        Returns:
            S3 URI string.
        """
        return f"s3://{self.bucket_name}/{key}"

    @staticmethod
    def _get_transfer_config() -> Any:
        """Get boto3 TransferConfig for multipart uploads."""
        from boto3.s3.transfer import TransferConfig

        return TransferConfig(
            multipart_threshold=MULTIPART_THRESHOLD,
            multipart_chunksize=MULTIPART_CHUNKSIZE,
            max_concurrency=MAX_CONCURRENCY,
        )

    def upload(self, file: FileInput, purpose: str | None = None) -> UploadResult:
        """Upload a file to S3 for use with Bedrock.

        Uses streaming upload with automatic multipart for large files.
        For FilePath sources, streams directly from disk without loading into memory.

        Args:
            file: The file to upload.
            purpose: Optional purpose (unused, kept for interface consistency).

        Returns:
            UploadResult with the S3 URI and metadata.

        Raises:
            TransientUploadError: For retryable errors (network, throttling).
            PermanentUploadError: For non-retryable errors (auth, validation).
        """
        import io

        try:
            client = self._get_client()
            transfer_config = self._get_transfer_config()
            file_path = _get_file_path(file)

            if file_path is not None:
                file_size = file_path.stat().st_size
                s3_key = self._generate_s3_key(file)

                logger.info(
                    f"Uploading file '{file.filename}' to S3 bucket "
                    f"'{self.bucket_name}' ({file_size} bytes, streaming)"
                )

                with open(file_path, "rb") as f:
                    client.upload_fileobj(
                        f,
                        self.bucket_name,
                        s3_key,
                        ExtraArgs={"ContentType": file.content_type},
                        Config=transfer_config,
                    )
            else:
                content = file.read()
                s3_key = self._generate_s3_key(file, content)

                logger.info(
                    f"Uploading file '{file.filename}' to S3 bucket "
                    f"'{self.bucket_name}' ({len(content)} bytes)"
                )

                client.upload_fileobj(
                    io.BytesIO(content),
                    self.bucket_name,
                    s3_key,
                    ExtraArgs={"ContentType": file.content_type},
                    Config=transfer_config,
                )

            s3_uri = self._build_s3_uri(s3_key)
            logger.info(f"Uploaded to S3: {s3_uri}")

            return UploadResult(
                file_id=s3_key,
                file_uri=s3_uri,
                content_type=file.content_type,
                expires_at=None,
                provider=self.provider_name,
            )
        except ImportError:
            raise
        except Exception as e:
            raise _classify_s3_error(e, file.filename) from e

    def delete(self, file_id: str) -> bool:
        """Delete an uploaded file from S3.

        Args:
            file_id: The S3 key to delete.

        Returns:
            True if deletion was successful, False otherwise.
        """
        try:
            client = self._get_client()
            client.delete_object(Bucket=self.bucket_name, Key=file_id)
            logger.info(f"Deleted S3 object: s3://{self.bucket_name}/{file_id}")
            return True
        except Exception as e:
            logger.warning(
                f"Failed to delete S3 object s3://{self.bucket_name}/{file_id}: {e}"
            )
            return False

    def get_file_info(self, file_id: str) -> dict[str, Any] | None:
        """Get information about an uploaded file.

        Args:
            file_id: The S3 key.

        Returns:
            Dictionary with file information, or None if not found.
        """
        try:
            client = self._get_client()
            response = client.head_object(Bucket=self.bucket_name, Key=file_id)
            return {
                "id": file_id,
                "uri": self._build_s3_uri(file_id),
                "content_type": response.get("ContentType"),
                "size": response.get("ContentLength"),
                "last_modified": response.get("LastModified"),
                "etag": response.get("ETag"),
            }
        except Exception as e:
            logger.debug(f"Failed to get S3 object info for {file_id}: {e}")
            return None

    def list_files(self) -> list[dict[str, Any]]:
        """List all uploaded files in the configured prefix.

        Returns:
            List of dictionaries with file information.
        """
        try:
            client = self._get_client()
            response = client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=self._prefix,
            )
            return [
                {
                    "id": obj["Key"],
                    "uri": self._build_s3_uri(obj["Key"]),
                    "size": obj.get("Size"),
                    "last_modified": obj.get("LastModified"),
                    "etag": obj.get("ETag"),
                }
                for obj in response.get("Contents", [])
            ]
        except Exception as e:
            logger.warning(f"Failed to list S3 objects: {e}")
            return []

    async def aupload(
        self, file: FileInput, purpose: str | None = None
    ) -> UploadResult:
        """Async upload a file to S3 for use with Bedrock.

        Uses streaming upload with automatic multipart for large files.
        For FilePath sources, streams directly from disk without loading into memory.

        Args:
            file: The file to upload.
            purpose: Optional purpose (unused, kept for interface consistency).

        Returns:
            UploadResult with the S3 URI and metadata.

        Raises:
            TransientUploadError: For retryable errors (network, throttling).
            PermanentUploadError: For non-retryable errors (auth, validation).
        """
        import io

        import aiofiles

        try:
            session = self._get_async_client()
            transfer_config = self._get_transfer_config()
            file_path = _get_file_path(file)

            if file_path is not None:
                file_size = file_path.stat().st_size
                s3_key = self._generate_s3_key(file)

                logger.info(
                    f"Uploading file '{file.filename}' to S3 bucket "
                    f"'{self.bucket_name}' ({file_size} bytes, streaming)"
                )

                async with session.client("s3", region_name=self._region) as client:
                    async with aiofiles.open(file_path, "rb") as f:
                        await client.upload_fileobj(
                            f,
                            self.bucket_name,
                            s3_key,
                            ExtraArgs={"ContentType": file.content_type},
                            Config=transfer_config,
                        )
            else:
                content = await file.aread()
                s3_key = self._generate_s3_key(file, content)

                logger.info(
                    f"Uploading file '{file.filename}' to S3 bucket "
                    f"'{self.bucket_name}' ({len(content)} bytes)"
                )

                async with session.client("s3", region_name=self._region) as client:
                    await client.upload_fileobj(
                        io.BytesIO(content),
                        self.bucket_name,
                        s3_key,
                        ExtraArgs={"ContentType": file.content_type},
                        Config=transfer_config,
                    )

            s3_uri = self._build_s3_uri(s3_key)
            logger.info(f"Uploaded to S3: {s3_uri}")

            return UploadResult(
                file_id=s3_key,
                file_uri=s3_uri,
                content_type=file.content_type,
                expires_at=None,
                provider=self.provider_name,
            )
        except ImportError:
            raise
        except Exception as e:
            raise _classify_s3_error(e, file.filename) from e

    async def adelete(self, file_id: str) -> bool:
        """Async delete an uploaded file from S3.

        Args:
            file_id: The S3 key to delete.

        Returns:
            True if deletion was successful, False otherwise.
        """
        try:
            session = self._get_async_client()
            async with session.client("s3", region_name=self._region) as client:
                await client.delete_object(Bucket=self.bucket_name, Key=file_id)
            logger.info(f"Deleted S3 object: s3://{self.bucket_name}/{file_id}")
            return True
        except Exception as e:
            logger.warning(
                f"Failed to delete S3 object s3://{self.bucket_name}/{file_id}: {e}"
            )
            return False
