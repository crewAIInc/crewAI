"""Cache for tracking uploaded files to avoid redundant uploads."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import hashlib
import logging
import threading

from crewai.utilities.files.content_types import (
    AudioFile,
    ImageFile,
    PDFFile,
    TextFile,
    VideoFile,
)


logger = logging.getLogger(__name__)

FileInput = AudioFile | ImageFile | PDFFile | TextFile | VideoFile


@dataclass
class CachedUpload:
    """Represents a cached file upload.

    Attributes:
        file_id: Provider-specific file identifier.
        provider: Name of the provider.
        file_uri: Optional URI for accessing the file.
        content_type: MIME type of the uploaded file.
        uploaded_at: When the file was uploaded.
        expires_at: When the upload expires (if applicable).
    """

    file_id: str
    provider: str
    file_uri: str | None
    content_type: str
    uploaded_at: datetime
    expires_at: datetime | None = None

    def is_expired(self) -> bool:
        """Check if this cached upload has expired.

        Returns:
            True if expired, False otherwise.
        """
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) >= self.expires_at


@dataclass
class UploadCache:
    """Thread-safe cache for tracking uploaded files.

    Uses file content hash and provider as composite key to avoid
    uploading the same file multiple times.

    Attributes:
        default_ttl: Default time-to-live for cached entries.
    """

    default_ttl: timedelta = field(default_factory=lambda: timedelta(hours=24))
    _cache: dict[tuple[str, str], CachedUpload] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _compute_hash(self, file: FileInput) -> str:
        """Compute a hash of file content for cache key.

        Args:
            file: The file to hash.

        Returns:
            SHA-256 hash of the file content.
        """
        content = file.source.read()
        return hashlib.sha256(content).hexdigest()

    def get(self, file: FileInput, provider: str) -> CachedUpload | None:
        """Get a cached upload for a file.

        Args:
            file: The file to look up.
            provider: The provider name.

        Returns:
            Cached upload if found and not expired, None otherwise.
        """
        file_hash = self._compute_hash(file)
        key = (file_hash, provider)

        with self._lock:
            cached = self._cache.get(key)
            if cached is None:
                return None

            if cached.is_expired():
                del self._cache[key]
                return None

            return cached

    def get_by_hash(self, file_hash: str, provider: str) -> CachedUpload | None:
        """Get a cached upload by file hash.

        Args:
            file_hash: Hash of the file content.
            provider: The provider name.

        Returns:
            Cached upload if found and not expired, None otherwise.
        """
        key = (file_hash, provider)

        with self._lock:
            cached = self._cache.get(key)
            if cached is None:
                return None

            if cached.is_expired():
                del self._cache[key]
                return None

            return cached

    def set(
        self,
        file: FileInput,
        provider: str,
        file_id: str,
        file_uri: str | None = None,
        expires_at: datetime | None = None,
    ) -> CachedUpload:
        """Cache an uploaded file.

        Args:
            file: The file that was uploaded.
            provider: The provider name.
            file_id: Provider-specific file identifier.
            file_uri: Optional URI for accessing the file.
            expires_at: When the upload expires.

        Returns:
            The created cache entry.
        """
        file_hash = self._compute_hash(file)
        key = (file_hash, provider)

        now = datetime.now(timezone.utc)
        cached = CachedUpload(
            file_id=file_id,
            provider=provider,
            file_uri=file_uri,
            content_type=file.content_type,
            uploaded_at=now,
            expires_at=expires_at,
        )

        with self._lock:
            self._cache[key] = cached

        logger.debug(f"Cached upload: {file_id} for provider {provider}")
        return cached

    def set_by_hash(
        self,
        file_hash: str,
        content_type: str,
        provider: str,
        file_id: str,
        file_uri: str | None = None,
        expires_at: datetime | None = None,
    ) -> CachedUpload:
        """Cache an uploaded file by hash.

        Args:
            file_hash: Hash of the file content.
            content_type: MIME type of the file.
            provider: The provider name.
            file_id: Provider-specific file identifier.
            file_uri: Optional URI for accessing the file.
            expires_at: When the upload expires.

        Returns:
            The created cache entry.
        """
        key = (file_hash, provider)

        now = datetime.now(timezone.utc)
        cached = CachedUpload(
            file_id=file_id,
            provider=provider,
            file_uri=file_uri,
            content_type=content_type,
            uploaded_at=now,
            expires_at=expires_at,
        )

        with self._lock:
            self._cache[key] = cached

        logger.debug(f"Cached upload: {file_id} for provider {provider}")
        return cached

    def remove(self, file: FileInput, provider: str) -> bool:
        """Remove a cached upload.

        Args:
            file: The file to remove.
            provider: The provider name.

        Returns:
            True if entry was removed, False if not found.
        """
        file_hash = self._compute_hash(file)
        key = (file_hash, provider)

        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def remove_by_file_id(self, file_id: str, provider: str) -> bool:
        """Remove a cached upload by file ID.

        Args:
            file_id: The file ID to remove.
            provider: The provider name.

        Returns:
            True if entry was removed, False if not found.
        """
        with self._lock:
            for key, cached in list(self._cache.items()):
                if cached.file_id == file_id and cached.provider == provider:
                    del self._cache[key]
                    return True
            return False

    def clear_expired(self) -> int:
        """Remove all expired entries from the cache.

        Returns:
            Number of entries removed.
        """
        removed = 0

        with self._lock:
            for key in list(self._cache.keys()):
                if self._cache[key].is_expired():
                    del self._cache[key]
                    removed += 1

        if removed > 0:
            logger.debug(f"Cleared {removed} expired cache entries")

        return removed

    def clear(self) -> int:
        """Clear all entries from the cache.

        Returns:
            Number of entries cleared.
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()

        if count > 0:
            logger.debug(f"Cleared {count} cache entries")

        return count

    def get_all_for_provider(self, provider: str) -> list[CachedUpload]:
        """Get all cached uploads for a provider.

        Args:
            provider: The provider name.

        Returns:
            List of cached uploads for the provider.
        """
        with self._lock:
            return [
                cached
                for (_, p), cached in self._cache.items()
                if p == provider and not cached.is_expired()
            ]

    def __len__(self) -> int:
        """Return the number of cached entries."""
        with self._lock:
            return len(self._cache)
