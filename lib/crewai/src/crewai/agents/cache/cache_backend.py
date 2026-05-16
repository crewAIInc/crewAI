"""Pluggable cache backend protocol for tool result caching."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class CacheBackend(Protocol):
    """Protocol for pluggable cache storage backends.

    Implementations must be thread-safe. The default in-memory backend
    uses a read-write lock; persistent backends (e.g. SQLite) should
    use their own appropriate locking strategy.
    """

    def get(self, key: str) -> Any | None:
        """Retrieve a cached value by key.

        Args:
            key: The cache key.

        Returns:
            The cached value, or None if the key is not present.
        """
        ...

    def set(self, key: str, value: Any) -> None:
        """Store a value in the cache.

        Args:
            key: The cache key.
            value: The value to cache.
        """
        ...

    def claim_if_absent(self, key: str, sentinel: Any) -> tuple[bool, Any | None]:
        """Atomically read the cache and write a sentinel if the key is absent.

        This is the core primitive for idempotent tool deduplication.

        Args:
            key: The cache key.
            sentinel: The sentinel value to write if the key is absent.

        Returns:
            (True, None) if the sentinel was written (caller owns the claim).
            (False, existing_value) if the key already existed.
        """
        ...
