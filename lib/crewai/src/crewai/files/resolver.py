"""FileResolver for deciding file delivery method and managing uploads."""

import asyncio
import base64
from dataclasses import dataclass, field
import hashlib
import logging

from crewai.files.content_types import (
    AudioFile,
    File,
    ImageFile,
    PDFFile,
    TextFile,
    VideoFile,
)
from crewai.files.metrics import measure_operation
from crewai.files.processing.constraints import (
    ProviderConstraints,
    get_constraints_for_provider,
)
from crewai.files.resolved import (
    FileReference,
    InlineBase64,
    InlineBytes,
    ResolvedFile,
)
from crewai.files.upload_cache import CachedUpload, UploadCache
from crewai.files.uploaders import UploadResult, get_uploader
from crewai.files.uploaders.base import FileUploader


logger = logging.getLogger(__name__)

FileInput = AudioFile | File | ImageFile | PDFFile | TextFile | VideoFile

UPLOAD_MAX_RETRIES = 3
UPLOAD_RETRY_DELAY_BASE = 2


@dataclass
class FileContext:
    """Cached file metadata to avoid redundant reads.

    Attributes:
        content: Raw file bytes.
        size: Size of the file in bytes.
        content_hash: SHA-256 hash of the file content.
        content_type: MIME type of the file.
    """

    content: bytes
    size: int
    content_hash: str
    content_type: str


@dataclass
class FileResolverConfig:
    """Configuration for FileResolver.

    Attributes:
        prefer_upload: If True, prefer uploading over inline for supported providers.
        upload_threshold_bytes: Size threshold above which to use upload.
            If None, uses provider-specific threshold.
        use_bytes_for_bedrock: If True, use raw bytes instead of base64 for Bedrock.
    """

    prefer_upload: bool = False
    upload_threshold_bytes: int | None = None
    use_bytes_for_bedrock: bool = True


@dataclass
class FileResolver:
    """Resolves files to their delivery format based on provider capabilities.

    Decides whether to use inline base64, raw bytes, or file upload based on:
    - Provider constraints and capabilities
    - File size
    - Configuration preferences

    Caches uploaded files to avoid redundant uploads.

    Attributes:
        config: Resolver configuration.
        upload_cache: Cache for tracking uploaded files.
    """

    config: FileResolverConfig = field(default_factory=FileResolverConfig)
    upload_cache: UploadCache | None = None
    _uploaders: dict[str, FileUploader] = field(default_factory=dict)

    def _build_file_context(self, file: FileInput) -> FileContext:
        """Build context by reading file once.

        Args:
            file: The file to build context for.

        Returns:
            FileContext with cached metadata.
        """
        content = file.read()
        return FileContext(
            content=content,
            size=len(content),
            content_hash=hashlib.sha256(content).hexdigest(),
            content_type=file.content_type,
        )

    def resolve(self, file: FileInput, provider: str) -> ResolvedFile:
        """Resolve a file to its delivery format for a provider.

        Args:
            file: The file to resolve.
            provider: Provider name (e.g., "gemini", "anthropic", "openai").

        Returns:
            ResolvedFile representing the appropriate delivery format.
        """
        provider_lower = provider.lower()
        constraints = get_constraints_for_provider(provider)
        context = self._build_file_context(file)

        should_upload = self._should_upload(
            file, provider_lower, constraints, context.size
        )

        if should_upload:
            resolved = self._resolve_via_upload(file, provider_lower, context)
            if resolved is not None:
                return resolved

        return self._resolve_inline(file, provider_lower, context)

    def resolve_files(
        self,
        files: dict[str, FileInput],
        provider: str,
    ) -> dict[str, ResolvedFile]:
        """Resolve multiple files for a provider.

        Args:
            files: Dictionary mapping names to file inputs.
            provider: Provider name.

        Returns:
            Dictionary mapping names to resolved files.
        """
        return {name: self.resolve(file, provider) for name, file in files.items()}

    def _should_upload(
        self,
        file: FileInput,
        provider: str,
        constraints: ProviderConstraints | None,
        file_size: int,
    ) -> bool:
        """Determine if a file should be uploaded rather than inlined.

        Args:
            file: The file to check.
            provider: Provider name.
            constraints: Provider constraints.
            file_size: Size of the file in bytes.

        Returns:
            True if the file should be uploaded, False otherwise.
        """
        if constraints is None or not constraints.supports_file_upload:
            return False

        if self.config.prefer_upload:
            return True

        threshold = self.config.upload_threshold_bytes
        if threshold is None and constraints is not None:
            threshold = constraints.file_upload_threshold_bytes

        if threshold is not None and file_size > threshold:
            return True

        return False

    def _resolve_via_upload(
        self,
        file: FileInput,
        provider: str,
        context: FileContext,
    ) -> ResolvedFile | None:
        """Resolve a file by uploading it.

        Args:
            file: The file to upload.
            provider: Provider name.
            context: Pre-computed file context.

        Returns:
            FileReference if upload succeeds, None otherwise.
        """
        if self.upload_cache is not None:
            cached = self.upload_cache.get_by_hash(context.content_hash, provider)
            if cached is not None:
                logger.debug(
                    f"Using cached upload for {file.filename}: {cached.file_id}"
                )
                return FileReference(
                    content_type=cached.content_type,
                    file_id=cached.file_id,
                    provider=cached.provider,
                    expires_at=cached.expires_at,
                    file_uri=cached.file_uri,
                )

        uploader = self._get_uploader(provider)
        if uploader is None:
            logger.debug(f"No uploader available for {provider}")
            return None

        result = self._upload_with_retry(uploader, file, provider, context.size)
        if result is None:
            return None

        if self.upload_cache is not None:
            self.upload_cache.set_by_hash(
                file_hash=context.content_hash,
                content_type=context.content_type,
                provider=provider,
                file_id=result.file_id,
                file_uri=result.file_uri,
                expires_at=result.expires_at,
            )

        return FileReference(
            content_type=result.content_type,
            file_id=result.file_id,
            provider=result.provider,
            expires_at=result.expires_at,
            file_uri=result.file_uri,
        )

    def _upload_with_retry(
        self,
        uploader: FileUploader,
        file: FileInput,
        provider: str,
        file_size: int,
    ) -> UploadResult | None:
        """Upload with exponential backoff retry.

        Args:
            uploader: The uploader to use.
            file: The file to upload.
            provider: Provider name for logging.
            file_size: Size of the file in bytes.

        Returns:
            UploadResult if successful, None otherwise.
        """
        import time

        from crewai.files.processing.exceptions import (
            PermanentUploadError,
            TransientUploadError,
        )

        last_error: Exception | None = None

        for attempt in range(UPLOAD_MAX_RETRIES):
            with measure_operation(
                "upload",
                filename=file.filename,
                provider=provider,
                size_bytes=file_size,
                attempt=attempt + 1,
            ) as metrics:
                try:
                    result = uploader.upload(file)
                    metrics.metadata["file_id"] = result.file_id
                    return result
                except PermanentUploadError as e:
                    metrics.metadata["error_type"] = "permanent"
                    logger.warning(
                        f"Non-retryable upload error for {file.filename}: {e}"
                    )
                    return None
                except TransientUploadError as e:
                    metrics.metadata["error_type"] = "transient"
                    last_error = e
                except Exception as e:
                    metrics.metadata["error_type"] = "unknown"
                    last_error = e

            if attempt < UPLOAD_MAX_RETRIES - 1:
                delay = UPLOAD_RETRY_DELAY_BASE**attempt
                logger.debug(
                    f"Retrying upload for {file.filename} in {delay}s (attempt {attempt + 1})"
                )
                time.sleep(delay)

        logger.warning(
            f"Upload failed for {file.filename} to {provider} after {UPLOAD_MAX_RETRIES} attempts: {last_error}"
        )
        return None

    def _resolve_inline(
        self,
        file: FileInput,
        provider: str,
        context: FileContext,
    ) -> ResolvedFile:
        """Resolve a file as inline content.

        Args:
            file: The file to resolve.
            provider: Provider name.
            context: Pre-computed file context.

        Returns:
            InlineBase64 or InlineBytes depending on provider.
        """
        if self.config.use_bytes_for_bedrock and "bedrock" in provider:
            return InlineBytes(
                content_type=context.content_type,
                data=context.content,
            )

        encoded = base64.b64encode(context.content).decode("ascii")
        return InlineBase64(
            content_type=context.content_type,
            data=encoded,
        )

    async def aresolve(self, file: FileInput, provider: str) -> ResolvedFile:
        """Async resolve a file to its delivery format for a provider.

        Args:
            file: The file to resolve.
            provider: Provider name (e.g., "gemini", "anthropic", "openai").

        Returns:
            ResolvedFile representing the appropriate delivery format.
        """
        provider_lower = provider.lower()
        constraints = get_constraints_for_provider(provider)
        context = self._build_file_context(file)

        should_upload = self._should_upload(
            file, provider_lower, constraints, context.size
        )

        if should_upload:
            resolved = await self._aresolve_via_upload(file, provider_lower, context)
            if resolved is not None:
                return resolved

        return self._resolve_inline(file, provider_lower, context)

    async def aresolve_files(
        self,
        files: dict[str, FileInput],
        provider: str,
        max_concurrency: int = 10,
    ) -> dict[str, ResolvedFile]:
        """Async resolve multiple files in parallel.

        Args:
            files: Dictionary mapping names to file inputs.
            provider: Provider name.
            max_concurrency: Maximum number of concurrent resolutions.

        Returns:
            Dictionary mapping names to resolved files.
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def resolve_one(name: str, file: FileInput) -> tuple[str, ResolvedFile]:
            async with semaphore:
                resolved = await self.aresolve(file, provider)
                return name, resolved

        tasks = [resolve_one(n, f) for n, f in files.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output: dict[str, ResolvedFile] = {}
        for result in results:
            if isinstance(result, BaseException):
                logger.error(f"Resolution failed: {result}")
                continue
            name, resolved = result
            output[name] = resolved

        return output

    async def _aresolve_via_upload(
        self,
        file: FileInput,
        provider: str,
        context: FileContext,
    ) -> ResolvedFile | None:
        """Async resolve a file by uploading it.

        Args:
            file: The file to upload.
            provider: Provider name.
            context: Pre-computed file context.

        Returns:
            FileReference if upload succeeds, None otherwise.
        """
        if self.upload_cache is not None:
            cached = await self.upload_cache.aget_by_hash(
                context.content_hash, provider
            )
            if cached is not None:
                logger.debug(
                    f"Using cached upload for {file.filename}: {cached.file_id}"
                )
                return FileReference(
                    content_type=cached.content_type,
                    file_id=cached.file_id,
                    provider=cached.provider,
                    expires_at=cached.expires_at,
                    file_uri=cached.file_uri,
                )

        uploader = self._get_uploader(provider)
        if uploader is None:
            logger.debug(f"No uploader available for {provider}")
            return None

        result = await self._aupload_with_retry(uploader, file, provider, context.size)
        if result is None:
            return None

        if self.upload_cache is not None:
            await self.upload_cache.aset_by_hash(
                file_hash=context.content_hash,
                content_type=context.content_type,
                provider=provider,
                file_id=result.file_id,
                file_uri=result.file_uri,
                expires_at=result.expires_at,
            )

        return FileReference(
            content_type=result.content_type,
            file_id=result.file_id,
            provider=result.provider,
            expires_at=result.expires_at,
            file_uri=result.file_uri,
        )

    async def _aupload_with_retry(
        self,
        uploader: FileUploader,
        file: FileInput,
        provider: str,
        file_size: int,
    ) -> UploadResult | None:
        """Async upload with exponential backoff retry.

        Args:
            uploader: The uploader to use.
            file: The file to upload.
            provider: Provider name for logging.
            file_size: Size of the file in bytes.

        Returns:
            UploadResult if successful, None otherwise.
        """
        from crewai.files.processing.exceptions import (
            PermanentUploadError,
            TransientUploadError,
        )

        last_error: Exception | None = None

        for attempt in range(UPLOAD_MAX_RETRIES):
            with measure_operation(
                "upload",
                filename=file.filename,
                provider=provider,
                size_bytes=file_size,
                attempt=attempt + 1,
            ) as metrics:
                try:
                    result = await uploader.aupload(file)
                    metrics.metadata["file_id"] = result.file_id
                    return result
                except PermanentUploadError as e:
                    metrics.metadata["error_type"] = "permanent"
                    logger.warning(
                        f"Non-retryable upload error for {file.filename}: {e}"
                    )
                    return None
                except TransientUploadError as e:
                    metrics.metadata["error_type"] = "transient"
                    last_error = e
                except Exception as e:
                    metrics.metadata["error_type"] = "unknown"
                    last_error = e

            if attempt < UPLOAD_MAX_RETRIES - 1:
                delay = UPLOAD_RETRY_DELAY_BASE**attempt
                logger.debug(
                    f"Retrying upload for {file.filename} in {delay}s (attempt {attempt + 1})"
                )
                await asyncio.sleep(delay)

        logger.warning(
            f"Upload failed for {file.filename} to {provider} after {UPLOAD_MAX_RETRIES} attempts: {last_error}"
        )
        return None

    def _get_uploader(self, provider: str) -> FileUploader | None:
        """Get or create an uploader for a provider.

        Args:
            provider: Provider name.

        Returns:
            FileUploader instance or None if not available.
        """
        if provider not in self._uploaders:
            uploader = get_uploader(provider)
            if uploader is not None:
                self._uploaders[provider] = uploader
            else:
                return None

        return self._uploaders.get(provider)

    def get_cached_uploads(self, provider: str) -> list[CachedUpload]:
        """Get all cached uploads for a provider.

        Args:
            provider: Provider name.

        Returns:
            List of cached uploads.
        """
        if self.upload_cache is None:
            return []
        return self.upload_cache.get_all_for_provider(provider)

    def clear_cache(self) -> None:
        """Clear the upload cache."""
        if self.upload_cache is not None:
            self.upload_cache.clear()


def create_resolver(
    provider: str | None = None,
    prefer_upload: bool = False,
    upload_threshold_bytes: int | None = None,
    enable_cache: bool = True,
) -> FileResolver:
    """Create a configured FileResolver.

    Args:
        provider: Optional provider name for provider-specific configuration.
        prefer_upload: Whether to prefer upload over inline.
        upload_threshold_bytes: Size threshold for using upload.
        enable_cache: Whether to enable upload caching.

    Returns:
        Configured FileResolver instance.
    """
    config = FileResolverConfig(
        prefer_upload=prefer_upload,
        upload_threshold_bytes=upload_threshold_bytes,
    )

    cache = UploadCache() if enable_cache else None

    return FileResolver(config=config, upload_cache=cache)
