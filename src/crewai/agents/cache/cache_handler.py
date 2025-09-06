"""Cache handler for tool usage results."""

from typing import Any

from pydantic import BaseModel, PrivateAttr


class CacheHandler(BaseModel):
    """Handles caching of tool execution results.

    Provides in-memory caching for tool outputs based on tool name and input.

    Notes:
        - TODO: Make thread-safe.
    """

    _cache: dict[str, Any] = PrivateAttr(default_factory=dict)

    def add(self, tool: str, input: str, output: Any) -> None:
        """Add a tool result to the cache.

        Args:
            tool: Name of the tool.
            input: Input string used for the tool.
            output: Output result from tool execution.

        Notes:
            - TODO: Rename 'input' parameter to avoid shadowing builtin.
        """
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
        return self._cache.get(f"{tool}-{input}")
