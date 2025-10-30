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
