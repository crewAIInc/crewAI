"""Cache for tracking uploaded files using aiocache or ValkeyCache."""

from __future__ import annotations

import asyncio
import atexit
import builtins
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import logging
from typing import TYPE_CHECKING, Any, Protocol

from aiocache import Cache  # type: ignore[import-untyped]
from aiocache.serializers import PickleSerializer  # type: ignore[import-untyped]
from crewai.utilities.cache_config import parse_cache_url

from crewai_files.core.constants import DEFAULT_MAX_CACHE_ENTRIES, DEFAULT_TTL_SECONDS
from crewai_files.uploaders.factory import ProviderType


if TYPE_CHECKING:
    from crewai_files.core.types import FileInput

logger = logging.getLogger(__name__)


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
    provider: ProviderType
    file_uri: str | None
    content_type: str
    uploaded_at: datetime
    expires_at: datetime | None = None

    def is_expired(self) -> bool:
        """Check if this cached upload has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) >= self.expires_at

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "file_id": self.file_id,
            "provider": self.provider,
            "file_uri": self.file_uri,
            "content_type": self.content_type,
            "uploaded_at": self.uploaded_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CachedUpload:
        """Deserialize from a dict."""
        return cls(
            file_id=data["file_id"],
            provider=data["provider"],
            file_uri=data.get("file_uri"),
            content_type=data["content_type"],
            uploaded_at=datetime.fromisoformat(data["uploaded_at"]),
            expires_at=(
                datetime.fromisoformat(data["expires_at"])
                if data.get("expires_at")
                else None
            ),
        )


def _make_key(file_hash: str, provider: str) -> str:
    """Create a cache key from file hash and provider."""
    return f"upload:{provider}:{file_hash}"


def _compute_file_hash_streaming(chunks: Iterator[bytes]) -> str:
    """Compute SHA-256 hash from streaming chunks."""
    hasher = hashlib.sha256()
    for chunk in chunks:
        hasher.update(chunk)
    return hasher.hexdigest()


def _compute_file_hash(file: FileInput) -> str:
    """Compute SHA-256 hash of file content."""
    from crewai_files.core.sources import FilePath

    source = file._file_source
    if isinstance(source, FilePath):
        return _compute_file_hash_streaming(source.read_chunks(chunk_size=1024 * 1024))
    content = file.read()
    return hashlib.sha256(content).hexdigest()


class CacheBackend(Protocol):
    """Protocol for cache backends used by UploadCache."""

    async def get(self, key: str) -> CachedUpload | None: ...
    async def set(self, key: str, value: CachedUpload, ttl: int) -> None: ...
    async def delete(self, key: str) -> bool: ...


class AiocacheBackend:
    """Cache backend backed by aiocache (memory or Redis)."""

    def __init__(self, cache: Cache) -> None:  # type: ignore[no-any-unimported]
        self._cache = cache

    async def get(self, key: str) -> CachedUpload | None:
        result = await self._cache.get(key)
        if isinstance(result, CachedUpload):
            return result
        return None

    async def set(self, key: str, value: CachedUpload, ttl: int) -> None:
        await self._cache.set(key, value, ttl=ttl)

    async def delete(self, key: str) -> bool:
        result = await self._cache.delete(key)
        return bool(result > 0 if isinstance(result, int) else result)


class ValkeyCacheBackend:
    """Cache backend backed by ValkeyCache (JSON serialization)."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: str | None = None,
        default_ttl: int | None = None,
    ) -> None:
        from crewai.memory.storage.valkey_cache import ValkeyCache

        self._cache = ValkeyCache(
            host=host, port=port, db=db, password=password, default_ttl=default_ttl
        )

    async def get(self, key: str) -> CachedUpload | None:
        data = await self._cache.get(key)
        if data is None:
            return None
        try:
            return CachedUpload.from_dict(data)
        except (KeyError, ValueError) as e:
            logger.warning(f"Failed to deserialize cached upload: {e}")
            return None

    async def set(self, key: str, value: CachedUpload, ttl: int) -> None:
        await self._cache.set(key, value.to_dict(), ttl=ttl)

    async def delete(self, key: str) -> bool:
        await self._cache.delete(key)
        return True  # ValkeyCache.delete is void


class UploadCache:
    """Async cache for tracking uploaded files.

    Supports in-memory caching by default, with optional Redis or Valkey backend
    for distributed setups.

    Attributes:
        ttl: Default time-to-live in seconds for cached entries.
        namespace: Cache namespace for isolation.
    """

    def __init__(
        self,
        ttl: int = DEFAULT_TTL_SECONDS,
        namespace: str = "crewai_uploads",
        cache_type: str = "memory",
        max_entries: int | None = DEFAULT_MAX_CACHE_ENTRIES,
        **cache_kwargs: Any,
    ) -> None:
        """Initialize the upload cache.

        Args:
            ttl: Default TTL in seconds.
            namespace: Cache namespace.
            cache_type: Backend type ("memory", "redis", or "valkey").
            max_entries: Maximum cache entries (None for unlimited).
            **cache_kwargs: Additional args for cache backend.
        """
        self.ttl = ttl
        self.namespace = namespace
        self.max_entries = max_entries
        self._provider_keys: dict[ProviderType, set[str]] = {}
        self._key_access_order: list[str] = []

        self._backend: CacheBackend = self._create_backend(
            cache_type, namespace, ttl, **cache_kwargs
        )

    @staticmethod
    def _create_backend(
        cache_type: str,
        namespace: str,
        ttl: int,
        **cache_kwargs: Any,
    ) -> CacheBackend:
        """Create the appropriate cache backend."""
        if cache_type == "valkey":
            conn = parse_cache_url() or {}
            return ValkeyCacheBackend(
                host=cache_kwargs.get("host", conn.get("host", "localhost")),
                port=cache_kwargs.get("port", conn.get("port", 6379)),
                db=cache_kwargs.get("db", conn.get("db", 0)),
                password=cache_kwargs.get("password", conn.get("password")),
                default_ttl=ttl,
            )
        if cache_type == "redis":
            return AiocacheBackend(
                Cache(
                    Cache.REDIS,
                    serializer=PickleSerializer(),
                    namespace=namespace,
                    **cache_kwargs,
                )
            )
        return AiocacheBackend(
            Cache(serializer=PickleSerializer(), namespace=namespace)
        )

    def _track_key(self, provider: ProviderType, key: str) -> None:
        """Track a key for a provider (for cleanup) and access order."""
        if provider not in self._provider_keys:
            self._provider_keys[provider] = set()
        self._provider_keys[provider].add(key)
        if key in self._key_access_order:
            self._key_access_order.remove(key)
        self._key_access_order.append(key)

    def _untrack_key(self, provider: ProviderType, key: str) -> None:
        """Remove key tracking for a provider."""
        if provider in self._provider_keys:
            self._provider_keys[provider].discard(key)
        if key in self._key_access_order:
            self._key_access_order.remove(key)

    async def _evict_if_needed(self) -> int:
        """Evict oldest entries if limit exceeded.

        Returns:
            Number of entries evicted.
        """
        if self.max_entries is None:
            return 0
        current_count = len(self)
        if current_count < self.max_entries:
            return 0
        to_evict = max(1, self.max_entries // 10)
        return await self._evict_oldest(to_evict)

    async def _evict_oldest(self, count: int) -> int:
        """Evict the oldest entries from the cache.

        Args:
            count: Number of entries to evict.

        Returns:
            Number of entries actually evicted.
        """
        evicted = 0
        keys_to_evict = self._key_access_order[:count]
        for key in keys_to_evict:
            await self._backend.delete(key)
            self._key_access_order.remove(key)
            for provider_keys in self._provider_keys.values():
                provider_keys.discard(key)
            evicted += 1
        if evicted > 0:
            logger.debug(f"Evicted {evicted} oldest cache entries")
        return evicted

    # ------------------------------------------------------------------
    # Async public API
    # ------------------------------------------------------------------

    async def aget(
        self, file: FileInput, provider: ProviderType
    ) -> CachedUpload | None:
        """Get a cached upload for a file."""
        file_hash = _compute_file_hash(file)
        return await self.aget_by_hash(file_hash, provider)

    async def aget_by_hash(
        self, file_hash: str, provider: ProviderType
    ) -> CachedUpload | None:
        """Get a cached upload by file hash.

        Args:
            file_hash: Hash of the file content.
            provider: The provider name.

        Returns:
            Cached upload if found and not expired, None otherwise.
        """
        key = _make_key(file_hash, provider)
        result = await self._backend.get(key)
        if result is None:
            return None
        if result.is_expired():
            await self._backend.delete(key)
            self._untrack_key(provider, key)
            return None
        return result

    async def aset(
        self,
        file: FileInput,
        provider: ProviderType,
        file_id: str,
        file_uri: str | None = None,
        expires_at: datetime | None = None,
    ) -> CachedUpload:
        """Cache an uploaded file."""
        file_hash = _compute_file_hash(file)
        return await self.aset_by_hash(
            file_hash=file_hash,
            content_type=file.content_type,
            provider=provider,
            file_id=file_id,
            file_uri=file_uri,
            expires_at=expires_at,
        )

    async def aset_by_hash(
        self,
        file_hash: str,
        content_type: str,
        provider: ProviderType,
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
        await self._evict_if_needed()
        key = _make_key(file_hash, provider)
        now = datetime.now(timezone.utc)

        cached = CachedUpload(
            file_id=file_id,
            provider=provider,
            file_uri=file_uri,
            content_type=content_type,
            uploaded_at=now,
            expires_at=expires_at,
        )

        ttl = self.ttl
        if expires_at is not None:
            ttl = max(0, int((expires_at - now).total_seconds()))

        await self._backend.set(key, cached, ttl=ttl)
        self._track_key(provider, key)
        logger.debug(f"Cached upload: {file_id} for provider {provider}")
        return cached

    async def aremove(self, file: FileInput, provider: ProviderType) -> bool:
        """Remove a cached upload.

        Args:
            file: The file to remove.
            provider: The provider name.

        Returns:
            True if entry was removed, False if not found.
        """
        file_hash = _compute_file_hash(file)
        key = _make_key(file_hash, provider)
        removed = await self._backend.delete(key)
        if removed:
            self._untrack_key(provider, key)
        return removed

    async def aremove_by_file_id(self, file_id: str, provider: ProviderType) -> bool:
        """Remove a cached upload by file ID.

        Args:
            file_id: The file ID to remove.
            provider: The provider name.

        Returns:
            True if entry was removed, False if not found.
        """
        if provider not in self._provider_keys:
            return False
        for key in list(self._provider_keys[provider]):
            cached = await self._backend.get(key)
            if cached is not None and cached.file_id == file_id:
                await self._backend.delete(key)
                self._untrack_key(provider, key)
                return True
        return False

    async def aclear_expired(self) -> int:
        """Remove all expired entries from the cache.

        Returns:
            Number of entries removed.
        """
        removed = 0
        for provider, keys in list(self._provider_keys.items()):
            for key in list(keys):
                cached = await self._backend.get(key)
                if cached is None or cached.is_expired():
                    await self._backend.delete(key)
                    self._untrack_key(provider, key)
                    removed += 1
        if removed > 0:
            logger.debug(f"Cleared {removed} expired cache entries")
        return removed

    async def aclear(self) -> int:
        """Clear all entries from the cache.

        Returns:
            Number of entries cleared.
        """
        count = sum(len(keys) for keys in self._provider_keys.values())
        # Delete all tracked keys individually (works for all backends)
        for keys in self._provider_keys.values():
            for key in keys:
                await self._backend.delete(key)
        self._provider_keys.clear()
        self._key_access_order.clear()
        if count > 0:
            logger.debug(f"Cleared {count} cache entries")
        return count

    async def aget_all_for_provider(self, provider: ProviderType) -> list[CachedUpload]:
        """Get all cached uploads for a provider.

        Args:
            provider: The provider name.

        Returns:
            List of cached uploads for the provider.
        """
        if provider not in self._provider_keys:
            return []
        results: list[CachedUpload] = []
        for key in list(self._provider_keys[provider]):
            cached = await self._backend.get(key)
            if cached is not None and not cached.is_expired():
                results.append(cached)
        return results

    # ------------------------------------------------------------------
    # Sync wrappers
    # ------------------------------------------------------------------

    @staticmethod
    def _run_sync(coro: Any) -> Any:
        """Run an async coroutine from sync context without blocking event loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            return future.result(timeout=30)
        return asyncio.run(coro)

    def get(self, file: FileInput, provider: ProviderType) -> CachedUpload | None:
        """Sync wrapper for aget."""
        result: CachedUpload | None = self._run_sync(self.aget(file, provider))
        return result

    def get_by_hash(
        self, file_hash: str, provider: ProviderType
    ) -> CachedUpload | None:
        """Sync wrapper for aget_by_hash."""
        result: CachedUpload | None = self._run_sync(
            self.aget_by_hash(file_hash, provider)
        )
        return result

    def set(
        self,
        file: FileInput,
        provider: ProviderType,
        file_id: str,
        file_uri: str | None = None,
        expires_at: datetime | None = None,
    ) -> CachedUpload:
        """Sync wrapper for aset."""
        result: CachedUpload = self._run_sync(
            self.aset(file, provider, file_id, file_uri, expires_at)
        )
        return result

    def set_by_hash(
        self,
        file_hash: str,
        content_type: str,
        provider: ProviderType,
        file_id: str,
        file_uri: str | None = None,
        expires_at: datetime | None = None,
    ) -> CachedUpload:
        """Sync wrapper for aset_by_hash."""
        result: CachedUpload = self._run_sync(
            self.aset_by_hash(
                file_hash, content_type, provider, file_id, file_uri, expires_at
            )
        )
        return result

    def remove(self, file: FileInput, provider: ProviderType) -> bool:
        """Sync wrapper for aremove."""
        result: bool = self._run_sync(self.aremove(file, provider))
        return result

    def remove_by_file_id(self, file_id: str, provider: ProviderType) -> bool:
        """Sync wrapper for aremove_by_file_id."""
        result: bool = self._run_sync(self.aremove_by_file_id(file_id, provider))
        return result

    def clear_expired(self) -> int:
        """Sync wrapper for aclear_expired."""
        result: int = self._run_sync(self.aclear_expired())
        return result

    def clear(self) -> int:
        """Sync wrapper for aclear."""
        result: int = self._run_sync(self.aclear())
        return result

    def get_all_for_provider(self, provider: ProviderType) -> list[CachedUpload]:
        """Sync wrapper for aget_all_for_provider."""
        result: list[CachedUpload] = self._run_sync(
            self.aget_all_for_provider(provider)
        )
        return result

    def __len__(self) -> int:
        """Return the number of cached entries."""
        return sum(len(keys) for keys in self._provider_keys.values())

    def get_providers(self) -> builtins.set[ProviderType]:
        """Get all provider names that have cached entries."""
        return builtins.set(self._provider_keys.keys())


_default_cache: UploadCache | None = None


def get_upload_cache(
    ttl: int = DEFAULT_TTL_SECONDS,
    namespace: str = "crewai_uploads",
    cache_type: str = "memory",
    **cache_kwargs: Any,
) -> UploadCache:
    """Get or create the default upload cache."""
    global _default_cache
    if _default_cache is None:
        _default_cache = UploadCache(
            ttl=ttl,
            namespace=namespace,
            cache_type=cache_type,
            **cache_kwargs,
        )
    return _default_cache


def reset_upload_cache() -> None:
    """Reset the default upload cache (useful for testing)."""
    global _default_cache
    if _default_cache is not None:
        _default_cache.clear()
    _default_cache = None


def _cleanup_on_exit() -> None:
    """Clean up uploaded files on process exit."""
    global _default_cache
    if _default_cache is None or len(_default_cache) == 0:
        return

    from crewai_files.cache.cleanup import cleanup_uploaded_files

    try:
        cleanup_uploaded_files(_default_cache)
    except Exception as e:
        logger.debug(f"Error during exit cleanup: {e}")


atexit.register(_cleanup_on_exit)
