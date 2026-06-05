"""Tools handler for managing tool execution, caching, and idempotency."""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, Field

from crewai.agents.cache.cache_handler import CacheHandler
from crewai.tools.cache_tools.cache_tools import CacheTools
from crewai.tools.tool_calling import InstructorToolCalling, ToolCalling


class ToolsHandler(BaseModel):
    """Callback handler for tool usage.

    Attributes:
        last_used_tool: The most recently used tool calling instance.
        cache: Optional cache handler for storing tool outputs.
    """

    cache: CacheHandler | None = Field(default=None)
    last_used_tool: ToolCalling | InstructorToolCalling | None = Field(default=None)

    # Idempotency store — prevents duplicate tool execution on task retry.
    # Scoped to the handler's lifetime (typically one agent lifetime).
    _idempotency_store: dict[str, str] = {}

    def _idempotency_key(self, tool_name: str, arguments: dict[str, object]) -> str:
        """Build a stable key from tool name and serialised arguments."""
        args_json = json.dumps(arguments, sort_keys=True, default=str)
        args_hash = hashlib.sha256(args_json.encode()).hexdigest()
        return f"{tool_name}:{args_hash}"

    def get_idempotent_result(
        self, tool_name: str, arguments: dict[str, object]
    ) -> str | None:
        """Return a previously stored result if this tool call already completed."""
        key = self._idempotency_key(tool_name, arguments)
        return self._idempotency_store.get(key)

    def set_idempotent_result(
        self, tool_name: str, arguments: dict[str, object], result: str
    ) -> None:
        """Store a tool result so that future retries can reuse it."""
        key = self._idempotency_key(tool_name, arguments)
        self._idempotency_store[key] = result

    def reset_idempotency_store(self) -> None:
        """Clear the idempotency store.

        Call this at the start of a fresh task execution to avoid stale
        entries from a previous unrelated task.
        """
        self._idempotency_store.clear()

    def on_tool_use(
        self,
        calling: ToolCalling | InstructorToolCalling,
        output: str,
        should_cache: bool = True,
    ) -> None:
        """Run when tool ends running.

        Args:
            calling: The tool calling instance.
            output: The output from the tool execution.
            should_cache: Whether to cache the tool output.
        """
        self.last_used_tool = calling

        # Store in idempotency store (independent of cache config)
        if calling.arguments:
            arguments: dict[str, object] = (
                calling.arguments
                if isinstance(calling.arguments, dict)
                else {}
            )
            self.set_idempotent_result(calling.tool_name, arguments, output)

        if self.cache and should_cache and calling.tool_name != CacheTools().name:
            input_str = ""
            if calling.arguments:
                if isinstance(calling.arguments, dict):
                    input_str = json.dumps(calling.arguments)
                else:
                    input_str = str(calling.arguments)

            self.cache.add(
                tool=calling.tool_name,
                input=input_str,
                output=output,
            )
