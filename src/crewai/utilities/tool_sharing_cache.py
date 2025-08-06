"""Tool sharing cache for optimizing multi-agent tool preparation."""

import hashlib
from typing import Dict, List, Optional, Any
from functools import lru_cache


class ToolSharingCache:
    """Cache for sharing prepared tools across agents to avoid redundant initialization."""

    def __init__(self, max_size: int = 128):
        """Initialize the tool sharing cache.

        Args:
            max_size: Maximum number of cached tool sets to retain
        """
        self._cache: Dict[str, List[Any]] = {}
        self._max_size = max_size
        self._usage_order: List[str] = []

    def _generate_cache_key(
        self,
        agent_id: str,
        task_id: str,
        tools: List[Any],
        allow_delegation: bool = False,
        allow_code_execution: bool = False,
        multimodal: bool = False,
        process_type: str = "sequential",
    ) -> str:
        """Generate a unique cache key for tool configuration.

        Args:
            agent_id: Unique identifier for the agent
            task_id: Unique identifier for the task
            tools: List of tools to be prepared
            allow_delegation: Whether delegation tools should be added
            allow_code_execution: Whether code execution tools should be added
            multimodal: Whether multimodal tools should be added
            process_type: The crew process type (sequential/hierarchical)

        Returns:
            A unique cache key for this tool configuration
        """
        tool_representations = []
        for i, tool in enumerate(tools):
            tool_name = getattr(tool, "name", None)
            if tool_name is None:
                tool_representations.append(f"tool_{i}_{id(tool)}")
            else:
                tool_representations.append(str(tool_name))

        tool_names_str = "|".join(sorted(tool_representations))
        config_str = f"{agent_id}|{task_id}|{tool_names_str}|{allow_delegation}|{allow_code_execution}|{multimodal}|{process_type}"
        return hashlib.sha256(config_str.encode("utf-8")).hexdigest()[:16]

    def get_tools(
        self,
        agent_id: str,
        task_id: str,
        tools: List[Any],
        allow_delegation: bool = False,
        allow_code_execution: bool = False,
        multimodal: bool = False,
        process_type: str = "sequential",
    ) -> Optional[List[Any]]:
        """Retrieve cached tools if available.

        Args:
            agent_id: Unique identifier for the agent
            task_id: Unique identifier for the task
            tools: List of tools to be prepared
            allow_delegation: Whether delegation tools should be added
            allow_code_execution: Whether code execution tools should be added
            multimodal: Whether multimodal tools should be added
            process_type: The crew process type

        Returns:
            Cached tools if found, None otherwise
        """
        cache_key = self._generate_cache_key(
            agent_id,
            task_id,
            tools,
            allow_delegation,
            allow_code_execution,
            multimodal,
            process_type,
        )

        if cache_key in self._cache:
            self._update_usage_order(cache_key)
            return self._cache[cache_key].copy()

        return None

    def store_tools(
        self,
        agent_id: str,
        task_id: str,
        tools: List[Any],
        prepared_tools: List[Any],
        allow_delegation: bool = False,
        allow_code_execution: bool = False,
        multimodal: bool = False,
        process_type: str = "sequential",
    ) -> None:
        """Store prepared tools in the cache.

        Args:
            agent_id: Unique identifier for the agent
            task_id: Unique identifier for the task
            tools: Original list of tools before preparation
            prepared_tools: The prepared tools to cache
            allow_delegation: Whether delegation tools were added
            allow_code_execution: Whether code execution tools were added
            multimodal: Whether multimodal tools were added
            process_type: The crew process type
        """
        cache_key = self._generate_cache_key(
            agent_id,
            task_id,
            tools,
            allow_delegation,
            allow_code_execution,
            multimodal,
            process_type,
        )

        if len(self._cache) >= self._max_size and cache_key not in self._cache:
            if self._max_size > 0 and len(self._usage_order) > 0:
                oldest_key = self._usage_order.pop(0)
                del self._cache[oldest_key]
            elif self._max_size == 0:
                return

        self._cache[cache_key] = prepared_tools.copy()
        self._update_usage_order(cache_key)

    def clear(self) -> None:
        """Clear all cached tools."""
        self._cache.clear()
        self._usage_order.clear()

    def size(self) -> int:
        """Return the current cache size."""
        return len(self._cache)

    def _update_usage_order(self, cache_key: str) -> None:
        """Update the usage order for LRU tracking."""
        if cache_key in self._usage_order:
            self._usage_order.remove(cache_key)
        self._usage_order.append(cache_key)

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics for monitoring."""
        return {
            "cache_size": len(self._cache),
            "max_size": self._max_size,
            "usage_order_length": len(self._usage_order),
        }


# Global tool sharing cache instance
_global_tool_cache: Optional[ToolSharingCache] = None


def get_tool_sharing_cache() -> ToolSharingCache:
    """Get the global tool sharing cache instance."""
    global _global_tool_cache
    if _global_tool_cache is None:
        _global_tool_cache = ToolSharingCache()
    return _global_tool_cache


@lru_cache(maxsize=64)
def _get_agent_tool_signature(
    agent_id: str, allow_delegation: bool, allow_code_execution: bool, multimodal: bool
) -> str:
    """Generate a signature for agent-specific tool configuration (cached for performance)."""
    return f"{agent_id}|{allow_delegation}|{allow_code_execution}|{multimodal}"


def should_use_tool_sharing(tools: List[Any]) -> bool:
    """Determine if tool sharing should be used based on tool characteristics.

    Args:
        tools: List of tools to evaluate

    Returns:
        True if tool sharing would be beneficial, False otherwise
    """
    return len(tools) >= 2
