"""FileResolver for deciding file delivery method and managing uploads."""

import asyncio
import base64
from dataclasses import dataclass, field
import hashlib
import logging

from crewai_files.cache.metrics import measure_operation
from crewai_files.cache.upload_cache import CachedUpload, UploadCache
from crewai_files.core.constants import UPLOAD_MAX_RETRIES, UPLOAD_RETRY_DELAY_BASE
from crewai_files.core.resolved import (
    FileReference,
    InlineBase64,
    InlineBytes,
    ResolvedFile,
    UrlReference,
)
from crewai_files.core.sources import FileUrl
from crewai_files.core.types import FileInput
from crewai_files.processing.constraints import (
    AudioConstraints,
    ImageConstraints,
    PDFConstraints,
    ProviderConstraints,
    VideoConstraints,
    get_constraints_for_provider,
)
from crewai_files.uploaders import UploadResult, get_uploader
from crewai_files.uploaders.base import FileUploader
from crewai_files.uploaders.factory import ProviderType


logger = logging.getLogger(__name__)


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

    @staticmethod
    def _build_file_context(file: FileInput) -> FileContext:
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

    @staticmethod
    def _is_url_source(file: FileInput) -> bool:
        """Check if file source is a URL.

        Args:
            file: The file to check.

        Returns:
            True if the file source is a FileUrl, False otherwise.
        """
        return isinstance(file._file_source, FileUrl)

    @staticmethod
    def _supports_url(constraints: ProviderConstraints | None) -> bool:
        """Check if provider supports URL references.

        Args:
            constraints: Provider constraints.

        Returns:
            True if the provider supports URL references, False otherwise.
        """
        return constraints is not None and constraints.supports_url_references

    @staticmethod
    def _resolve_as_url(file: FileInput) -> UrlReference:
        """Resolve a URL source as UrlReference.

        Args:
            file: The file with URL source.

        Returns:
            UrlReference with the URL and content type.
        """
        source = file._file_source
        if not isinstance(source, FileUrl):
            raise TypeError(f"Expected FileUrl source, got {type(source).__name__}")
        return UrlReference(
            content_type=file.content_type,
            url=source.url,
        )

    def resolve(self, file: FileInput, provider: ProviderType) -> ResolvedFile:
        """Resolve a file to its delivery format for a provider.

        Args:
            file: The file to resolve.
            provider: Provider name (e.g., "gemini", "anthropic", "openai").

        Returns:
            ResolvedFile representing the appropriate delivery format.
        """
        constraints = get_constraints_for_provider(provider)

        if self._is_url_source(file) and self._supports_url(constraints):
            return self._resolve_as_url(file)

        context = self._build_file_context(file)

        should_upload = self._should_upload(file, provider, constraints, context.size)

        if should_upload:
            resolved = self._resolve_via_upload(file, provider, context)
            if resolved is not None:
                return resolved

        return self._resolve_inline(file, provider, context)

    def resolve_files(
        self,
        files: dict[str, FileInput],
        provider: ProviderType,
    ) -> dict[str, ResolvedFile]:
        """Resolve multiple files for a provider.

        Args:
            files: Dictionary mapping names to file inputs.
            provider: Provider name.

        Returns:
            Dictionary mapping names to resolved files.
        """
        return {name: self.resolve(file, provider) for name, file in files.items()}

    @staticmethod
    def _get_type_constraint(
        content_type: str,
        constraints: ProviderConstraints,
    ) -> ImageConstraints | PDFConstraints | AudioConstraints | VideoConstraints | None:
        """Get type-specific constraint based on content type.

        Args:
            content_type: MIME type of the file.
            constraints: Provider constraints.

        Returns:
            Type-specific constraint or None if not found.
        """
        if content_type.startswith("image/"):
            return constraints.image
        if content_type == "application/pdf":
            return constraints.pdf
        if content_type.startswith("audio/"):
            return constraints.audio
        if content_type.startswith("video/"):
            return constraints.video
        return None

    def _should_upload(
        self,
        file: FileInput,
        provider: str,
        constraints: ProviderConstraints | None,
        file_size: int,
    ) -> bool:
        """Determine if a file should be uploaded rather than inlined.

        Uses type-specific constraints to make smarter decisions:
        - Checks if file exceeds type-specific inline size limits
        - Falls back to general threshold if no type-specific constraint

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

        content_type = file.content_type
        type_constraint = self._get_type_constraint(content_type, constraints)

        if type_constraint is not None:
            # Check if file exceeds type-specific inline limit
            if file_size > type_constraint.max_size_bytes:
                logger.debug(
                    f"File {file.filename} ({file_size}B) exceeds {content_type} "
                    f"inline limit ({type_constraint.max_size_bytes}B) for {provider}"
                )
                return True

        # Fall back to general threshold
        threshold = self.config.upload_threshold_bytes
        if threshold is None:
            threshold = constraints.file_upload_threshold_bytes

        if threshold is not None and file_size > threshold:
            return True

        return False

    def _resolve_via_upload(
        self,
        file: FileInput,
        provider: ProviderType,
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

    @staticmethod
    def _upload_with_retry(
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

        from crewai_files.processing.exceptions import (
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
            file: The file to resolve (used for logging).
            provider: Provider name.
            context: Pre-computed file context.

        Returns:
            InlineBase64 or InlineBytes depending on provider.
        """
        logger.debug(f"Resolving {file.filename} as inline for {provider}")
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

    async def aresolve(self, file: FileInput, provider: ProviderType) -> ResolvedFile:
        """Async resolve a file to its delivery format for a provider.

        Args:
            file: The file to resolve.
            provider: Provider name (e.g., "gemini", "anthropic", "openai").

        Returns:
            ResolvedFile representing the appropriate delivery format.
        """
        constraints = get_constraints_for_provider(provider)

        if self._is_url_source(file) and self._supports_url(constraints):
            return self._resolve_as_url(file)

        context = self._build_file_context(file)

        should_upload = self._should_upload(file, provider, constraints, context.size)

        if should_upload:
            resolved = await self._aresolve_via_upload(file, provider, context)
            if resolved is not None:
                return resolved

        return self._resolve_inline(file, provider, context)

    async def aresolve_files(
        self,
        files: dict[str, FileInput],
        provider: ProviderType,
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

        async def resolve_single(
            entry_key: str, input_file: FileInput
        ) -> tuple[str, ResolvedFile]:
            """Resolve a single file with semaphore limiting."""
            async with semaphore:
                entry_resolved = await self.aresolve(input_file, provider)
                return entry_key, entry_resolved

        tasks = [resolve_single(n, f) for n, f in files.items()]
        gather_results = await asyncio.gather(*tasks, return_exceptions=True)

        output: dict[str, ResolvedFile] = {}
        for item in gather_results:
            if isinstance(item, BaseException):
                logger.error(f"Resolution failed: {item}")
                continue
            key, resolved = item
            output[key] = resolved

        return output

    async def _aresolve_via_upload(
        self,
        file: FileInput,
        provider: ProviderType,
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

    @staticmethod
    async def _aupload_with_retry(
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
        from crewai_files.processing.exceptions import (
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

    def _get_uploader(self, provider: ProviderType) -> FileUploader | None:
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

    def get_cached_uploads(self, provider: ProviderType) -> list[CachedUpload]:
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
        provider: Optional provider name to load default threshold from constraints.
        prefer_upload: Whether to prefer upload over inline.
        upload_threshold_bytes: Size threshold for using upload. If None and
            provider is specified, uses provider's default threshold.
        enable_cache: Whether to enable upload caching.

    Returns:
        Configured FileResolver instance.
    """
    threshold = upload_threshold_bytes
    if threshold is None and provider is not None:
        constraints = get_constraints_for_provider(provider)
        if constraints is not None:
            threshold = constraints.file_upload_threshold_bytes

    config = FileResolverConfig(
        prefer_upload=prefer_upload,
        upload_threshold_bytes=threshold,
    )

    cache = UploadCache() if enable_cache else None

    return FileResolver(config=config, upload_cache=cache)
