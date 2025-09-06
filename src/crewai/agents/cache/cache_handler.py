"""Cache handler for storing and retrieving tool execution results.

This module provides a caching mechanism for tool outputs in the CrewAI framework,
allowing agents to reuse previous tool execution results when the same tool is
called with identical arguments.

Classes:
    CacheHandler: Manages the caching of tool execution results using an in-memory
        dictionary with serialized tool arguments as keys.
"""

import json
from typing import Any

from pydantic import BaseModel, PrivateAttr


class CacheHandler(BaseModel):
    """Callback handler for tool usage.


    Notes:
        TODO: Make thread-safe, currently not thread-safe.
    """

    _cache: dict[str, Any] = PrivateAttr(default_factory=dict)

    def add(self, tool: str, input_data: dict[str, Any] | None, output: str) -> None:
        """Add a tool execution result to the cache.

        Args:
            tool: The name of the tool.
            input_data: The input arguments for the tool.
            output: The output from the tool execution.
        """
        cache_key = json.dumps(input_data, sort_keys=True) if input_data else ""
        self._cache[f"{tool}-{cache_key}"] = output

    def read(self, tool: str, input_data: dict[str, Any] | None) -> str | None:
        """Read a tool execution result from the cache.

        Args:
            tool: The name of the tool.
            input_data: The input arguments for the tool.

        Returns:
            The cached output if found, None otherwise.
        """
        cache_key = json.dumps(input_data, sort_keys=True) if input_data else ""
        return self._cache.get(f"{tool}-{cache_key}")
