"""FileResolver for deciding file delivery method and managing uploads."""

import base64
from dataclasses import dataclass, field
import logging

from crewai.utilities.files.content_types import (
    AudioFile,
    File,
    ImageFile,
    PDFFile,
    TextFile,
    VideoFile,
)
from crewai.utilities.files.processing.constraints import (
    ProviderConstraints,
    get_constraints_for_provider,
)
from crewai.utilities.files.resolved import (
    FileReference,
    InlineBase64,
    InlineBytes,
    ResolvedFile,
)
from crewai.utilities.files.upload_cache import CachedUpload, UploadCache
from crewai.utilities.files.uploaders import get_uploader
from crewai.utilities.files.uploaders.base import FileUploader


logger = logging.getLogger(__name__)

FileInput = AudioFile | File | ImageFile | PDFFile | TextFile | VideoFile


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
        file_size = len(file.source.read())

        # Determine if we should use file upload
        should_upload = self._should_upload(
            file, provider_lower, constraints, file_size
        )

        if should_upload:
            resolved = self._resolve_via_upload(file, provider_lower)
            if resolved is not None:
                return resolved
            # Fall back to inline if upload fails

        # Use inline format
        return self._resolve_inline(file, provider_lower)

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
        # Check if provider supports file upload
        if constraints is None or not constraints.supports_file_upload:
            return False

        # If prefer_upload is set, always prefer upload
        if self.config.prefer_upload:
            return True

        # Check against size threshold
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
    ) -> ResolvedFile | None:
        """Resolve a file by uploading it.

        Args:
            file: The file to upload.
            provider: Provider name.

        Returns:
            FileReference if upload succeeds, None otherwise.
        """
        # Check cache first
        if self.upload_cache is not None:
            cached = self.upload_cache.get(file, provider)
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

        # Get or create uploader
        uploader = self._get_uploader(provider)
        if uploader is None:
            logger.debug(f"No uploader available for {provider}")
            return None

        try:
            result = uploader.upload(file)

            # Cache the result
            if self.upload_cache is not None:
                self.upload_cache.set(
                    file=file,
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

        except Exception as e:
            logger.warning(f"Failed to upload {file.filename} to {provider}: {e}")
            return None

    def _resolve_inline(self, file: FileInput, provider: str) -> ResolvedFile:
        """Resolve a file as inline content.

        Args:
            file: The file to resolve.
            provider: Provider name.

        Returns:
            InlineBase64 or InlineBytes depending on provider.
        """
        content = file.source.read()

        # Use raw bytes for Bedrock if configured
        if self.config.use_bytes_for_bedrock and "bedrock" in provider:
            return InlineBytes(
                content_type=file.content_type,
                data=content,
            )

        # Default to base64
        encoded = base64.b64encode(content).decode("ascii")
        return InlineBase64(
            content_type=file.content_type,
            data=encoded,
        )

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
