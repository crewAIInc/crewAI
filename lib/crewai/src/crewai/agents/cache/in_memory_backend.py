"""In-memory cache backend (default)."""

from __future__ import annotations

from typing import Any

from crewai.utilities.rw_lock import RWLock


class InMemoryCacheBackend:
    """Thread-safe in-memory cache backend.

    This is the default backend used by CacheHandler. It stores values in
    a plain dict guarded by a read-write lock, allowing concurrent reads
    with exclusive writes.

    Suitable for single-process use. For cross-process / distributed
    deduplication, use SQLiteCacheBackend or another persistent backend.
    """

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}
        self._lock = RWLock()

    def get(self, key: str) -> Any | None:
        with self._lock.r_locked():
            return self._cache.get(key)

    def set(self, key: str, value: Any) -> None:
        with self._lock.w_locked():
            self._cache[key] = value

    def claim_if_absent(self, key: str, sentinel: Any) -> tuple[bool, Any | None]:
        with self._lock.w_locked():
            if key in self._cache:
                return False, self._cache[key]
            self._cache[key] = sentinel
            return True, None
