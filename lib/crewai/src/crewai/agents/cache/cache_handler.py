"""Cache handler for tool usage results."""

from typing import Any

from pydantic import BaseModel, PrivateAttr

from crewai.utilities.rw_lock import RWLock


class CacheHandler(BaseModel):
    """Handles caching of tool execution results.

    Provides thread-safe in-memory caching for tool outputs based on tool name and input.
    Uses a read-write lock to allow concurrent reads while ensuring exclusive write access.

    Notes:
        - TODO: Rename 'input' parameter to avoid shadowing builtin.
    """

    _cache: dict[str, Any] = PrivateAttr(default_factory=dict)
    _lock: RWLock = PrivateAttr(default_factory=RWLock)

    def add(self, tool: str, input: str, output: Any) -> None:
        """Add a tool result to the cache.

        Args:
            tool: Name of the tool.
            input: Input string used for the tool.
            output: Output result from tool execution.

        Notes:
            - TODO: Rename 'input' parameter to avoid shadowing builtin.
        """
        with self._lock.w_locked():
            self._cache[f"{tool}-{input}"] = output

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
        with self._lock.r_locked():
            return self._cache.get(f"{tool}-{input}")

    def claim_if_absent(self, tool: str, input: str, sentinel: Any) -> tuple[bool, Any | None]:
        """Atomically read the cache and write a sentinel if absent.

        Returns:
            (True, None) if the sentinel was written (caller owns the claim).
            (False, existing_value) if the key already existed.
        """
        key = f"{tool}-{input}"
        with self._lock.w_locked():
            existing = self._cache.get(key)
            if existing is not None:
                return False, existing
            self._cache[key] = sentinel
            return True, None
