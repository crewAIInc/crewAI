"""Tool sharing cache for optimizing multi-agent tool preparation."""

import hashlib
from collections import OrderedDict
from typing import Dict, List, Optional, Any


class ToolSharingCache:
    """Cache for sharing prepared tools across agents to avoid redundant initialization."""

    # No need for class-level constants anymore since we simplified

    def __init__(self, max_size: int = 128):
        """Initialize the tool sharing cache with OrderedDict for O(1) LRU operations.

        Args:
            max_size: Maximum number of cached tool sets to retain
        """
        if max_size < 0:
            raise ValueError("max_size must be non-negative")
        self._cache: OrderedDict[str, List[Any]] = OrderedDict()
        self._max_size = max_size
        # Cache for tool identifiers to avoid recomputation
        self._tool_id_cache: Dict[int, str] = {}

    def _get_tool_identifier(self, tool: Any) -> str:
        """Generate a unique identifier for a Tool/BaseTool instance with caching.

        Uses all relevant Tool attributes to create a deterministic identifier:
        - name, description, func, args_schema, and other configuration

        Args:
            tool: A Tool or BaseTool instance

        Returns:
            A unique string identifier for the tool
        """
        # Use object id for cache lookup
        tool_obj_id = id(tool)

        # Return cached identifier if available
        if tool_obj_id in self._tool_id_cache:
            return self._tool_id_cache[tool_obj_id]

        # Build configuration efficiently
        config_parts = []

        # Get name once and reuse
        tool_name = getattr(tool, "name", None) or f"tool_{id(tool)}"
        if tool_name:
            config_parts.append(f"n:{tool_name}")

        # Add description hash if present (use hash to keep key small)
        description = getattr(tool, "description", None)
        if description:
            desc_str = (
                str(description) if not isinstance(description, str) else description
            )
            config_parts.append(
                f"d:{hashlib.blake2s(desc_str.encode()).hexdigest()[:8]}"
            )

        # Add function identity for Tool instances
        func = getattr(tool, "func", None)
        if func and callable(func):
            if hasattr(func, "__code__"):
                # Use function's code location for identity
                code = func.__code__
                func_id = f"{code.co_filename}#{code.co_firstlineno}"
                config_parts.append(
                    f"f:{hashlib.blake2s(func_id.encode()).hexdigest()[:8]}"
                )

        # Extract configuration from args_schema efficiently
        args_schema = getattr(tool, "args_schema", None)
        if args_schema:
            try:
                if hasattr(args_schema, "model_fields"):
                    defaults = []
                    for field_name, field_info in args_schema.model_fields.items():
                        if hasattr(field_info, "default"):
                            default = field_info.default
                            # Skip undefined markers efficiently
                            if (
                                default is not None
                                and str(type(default).__name__) != "PydanticUndefined"
                            ):
                                defaults.append(f"{field_name}={default}")

                    if defaults:
                        defaults_str = "&".join(defaults)
                        config_parts.append(
                            f"as:{hashlib.blake2s(defaults_str.encode()).hexdigest()[:8]}"
                        )
            except (TypeError, AttributeError):
                pass

        # Add behavioral attributes that affect tool execution
        result_as_answer = getattr(tool, "result_as_answer", False)
        if result_as_answer:
            config_parts.append("ra:1")

        max_usage = getattr(tool, "max_usage_count", None)
        if max_usage is not None:
            config_parts.append(f"mu:{max_usage}")

        # Generate final identifier
        if config_parts:
            # Use blake2s for better performance than md5/sha256
            identifier = hashlib.blake2s("|".join(config_parts).encode()).hexdigest()[
                :16
            ]
        else:
            identifier = f"id_{tool_obj_id}"

        result = f"{tool_name or 'tool'}:{identifier}"

        # Cache the result
        self._tool_id_cache[tool_obj_id] = result
        return result

    def _generate_cache_key(
        self,
        tools: List[Any],
        allow_delegation: bool = False,
        allow_code_execution: bool = False,
        multimodal: bool = False,
        process_type: str = "sequential",
    ) -> str:
        """Generate a unique cache key for tool configuration.

        Note: Removed agent_id and task_id to improve cache reuse across agents.

        Args:
            tools: List of tools to be prepared
            allow_delegation: Whether delegation tools should be added
            allow_code_execution: Whether code execution tools should be added
            multimodal: Whether multimodal tools should be added
            process_type: The crew process type (sequential/hierarchical)

        Returns:
            A unique cache key for this tool configuration
        """
        # Generate unique identifiers for each tool
        tool_identifiers = tuple(
            sorted(self._get_tool_identifier(tool) for tool in tools)
        )

        # Combine with configuration flags
        key_data = (
            tool_identifiers,
            allow_delegation,
            allow_code_execution,
            multimodal,
            process_type,
        )

        # Use blake2s for better performance than sha256
        key_str = str(key_data)
        return hashlib.blake2s(key_str.encode()).hexdigest()[:24]

    def get_tools(
        self,
        tools: List[Any],
        allow_delegation: bool = False,
        allow_code_execution: bool = False,
        multimodal: bool = False,
        process_type: str = "sequential",
    ) -> Optional[List[Any]]:
        """Retrieve cached tools if available.

        Note: Removed agent_id and task_id to improve cache reuse.

        Args:
            tools: List of tools to be prepared
            allow_delegation: Whether delegation tools should be added
            allow_code_execution: Whether code execution tools should be added
            multimodal: Whether multimodal tools should be added
            process_type: The crew process type

        Returns:
            Cached tools if found, None otherwise
        """
        cache_key = self._generate_cache_key(
            tools,
            allow_delegation,
            allow_code_execution,
            multimodal,
            process_type,
        )

        if cache_key in self._cache:
            # Move to end for LRU (OrderedDict maintains order)
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key].copy()

        return None

    def store_tools(
        self,
        tools: List[Any],
        prepared_tools: List[Any],
        allow_delegation: bool = False,
        allow_code_execution: bool = False,
        multimodal: bool = False,
        process_type: str = "sequential",
    ) -> None:
        """Store prepared tools in the cache.

        Note: Removed agent_id and task_id to improve cache reuse.

        Args:
            tools: Original list of tools before preparation
            prepared_tools: The prepared tools to cache
            allow_delegation: Whether delegation tools were added
            allow_code_execution: Whether code execution tools were added
            multimodal: Whether multimodal tools were added
            process_type: The crew process type
        """
        if self._max_size == 0:
            return

        cache_key = self._generate_cache_key(
            tools,
            allow_delegation,
            allow_code_execution,
            multimodal,
            process_type,
        )

        # Evict if needed - OrderedDict pops first (oldest) item
        if cache_key not in self._cache and len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)

        self._cache[cache_key] = prepared_tools.copy()
        # Move to end to mark as recently used
        self._cache.move_to_end(cache_key)

    def clear(self) -> None:
        """Clear all cached tools."""
        self._cache.clear()
        self._tool_id_cache.clear()

    def size(self) -> int:
        """Return the current cache size."""
        return len(self._cache)

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "tool_id_cache_size": len(self._tool_id_cache),
        }


# Global tool sharing cache instance
_global_tool_cache: Optional[ToolSharingCache] = None


def get_tool_sharing_cache() -> ToolSharingCache:
    """Get the global tool sharing cache instance."""
    global _global_tool_cache
    if _global_tool_cache is None:
        _global_tool_cache = ToolSharingCache()
    return _global_tool_cache


def should_use_tool_sharing(tools: List[Any]) -> bool:
    """Determine if tool sharing should be used based on tool characteristics.

    Args:
        tools: List of tools to evaluate

    Returns:
        True if tool sharing would be beneficial, False otherwise
    """
    return len(tools) > 0
