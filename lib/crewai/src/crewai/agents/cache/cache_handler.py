"""Cache handler for tool usage results."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, PrivateAttr

from crewai.agents.cache.cache_backend import CacheBackend
from crewai.agents.cache.in_memory_backend import InMemoryCacheBackend


class CacheHandler(BaseModel):
    """Handles caching of tool execution results.

    Delegates all storage to a pluggable :class:`CacheBackend`.
    The default backend is :class:`InMemoryCacheBackend` (single-process,
    thread-safe).  For cross-process / distributed deduplication pass a
    :class:`SQLiteCacheBackend` (or any other ``CacheBackend`` implementation).

    Args:
        backend: Optional cache backend.  When *None* an
            :class:`InMemoryCacheBackend` is created automatically.

    Notes:
        - TODO: Rename 'input' parameter to avoid shadowing builtin.
    """

    _backend: CacheBackend = PrivateAttr()

    def __init__(self, backend: CacheBackend | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._backend = backend if backend is not None else InMemoryCacheBackend()

    def add(self, tool: str, input: str, output: Any) -> None:
        """Add a tool result to the cache.

        Args:
            tool: Name of the tool.
            input: Input string used for the tool.
            output: Output result from tool execution.

        Notes:
            - TODO: Rename 'input' parameter to avoid shadowing builtin.
        """
        self._backend.set(f"{tool}-{input}", output)

    def read(self, tool: str, input: str) -> Any | None:
        """Retrieve a cached tool result.

        Args:
            tool: Name of the tool.
            input: Input string used for the tool.

        Returns:
            Cached result if found, None otherwise.

        Notes:
            - TODO: Rename 'input' parameter to avoid shadowing builtin.
        """
        return self._backend.get(f"{tool}-{input}")

    def claim_if_absent(self, tool: str, input: str, sentinel: Any) -> tuple[bool, Any | None]:
        """Atomically read the cache and write a sentinel if absent.

        Returns:
            (True, None) if the sentinel was written (caller owns the claim).
            (False, existing_value) if the key already existed.
        """
        return self._backend.claim_if_absent(f"{tool}-{input}", sentinel)
