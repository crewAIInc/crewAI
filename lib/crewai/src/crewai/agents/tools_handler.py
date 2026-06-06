"""Tools handler for managing tool execution, caching, and idempotency."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, PrivateAttr

from crewai.agents.cache.cache_handler import CacheHandler
from crewai.tools.cache_tools.cache_tools import CacheTools
from crewai.tools.tool_calling import InstructorToolCalling, ToolCalling
from crewai.utilities.string_utils import sanitize_tool_name

if TYPE_CHECKING:
    from crewai.utilities.idempotency_backend import IdempotencyBackend


class ToolsHandler(BaseModel):
    """Callback handler for tool usage.

    Attributes:
        last_used_tool: The most recently used tool calling instance.
        cache: Optional cache handler for storing tool outputs.
    """

    cache: CacheHandler | None = Field(default=None)
    last_used_tool: ToolCalling | InstructorToolCalling | None = Field(default=None)

    # ------------------------------------------------------------------
    # Idempotency — prevents duplicate tool side effects on task retry.
    # Uses a pluggable backend (default: MemoryIdempotencyBackend).
    # Inject a persistent backend for cross-process / cross-worker safety.
    # ------------------------------------------------------------------
    _idempotency_backend: IdempotencyBackend | None = PrivateAttr(default=None)

    def _get_backend(self) -> IdempotencyBackend:
        """Lazily initialise the default in-memory backend."""
        if self._idempotency_backend is None:
            from crewai.utilities.idempotency_backend import MemoryIdempotencyBackend

            self._idempotency_backend = MemoryIdempotencyBackend()
        return self._idempotency_backend

    def set_idempotency_backend(self, backend: IdempotencyBackend) -> None:
        """Replace the idempotency backend (e.g. with a Redis-backed one).

        Call once before any tool execution.
        """
        self._idempotency_backend = backend

    def _idempotency_key(self, tool_name: str, arguments: dict[str, object]) -> str:
        """Build a stable key from sanitised tool name and serialised arguments."""
        args_json = json.dumps(arguments, sort_keys=True, default=str)
        args_hash = hashlib.sha256(args_json.encode()).hexdigest()
        return f"{sanitize_tool_name(tool_name)}:{args_hash}"

    def get_idempotent_result(
        self, tool_name: str, arguments: dict[str, object]
    ) -> str | None:
        """Return a previously stored result if this tool call already completed."""
        key = self._idempotency_key(tool_name, arguments)
        return self._get_backend().get(key)

    def claim_idempotent_result(
        self, tool_name: str, arguments: dict[str, object]
    ) -> str | None:
        """Atomically try to claim the execution slot for this tool call.

        Returns the stored result if already completed, or ``None`` if the
        caller should proceed with execution (the claim has been recorded).

        Two concurrent callers for the same key will never both receive
        ``None`` — one will get the result and the other will get ``None``.
        """
        key = self._idempotency_key(tool_name, arguments)
        claimed, result = self._get_backend().claim(key)
        if claimed:
            return None  # caller must execute
        # Not claimed: either completed (result is the output) or in-progress
        # (result is None).  In the in-progress case we return None so the
        # caller falls through to normal execution — the race window is
        # narrowed to a brief in-progress interval.
        return result

    def set_idempotent_result(
        self, tool_name: str, arguments: dict[str, object], result: str
    ) -> None:
        """Store a tool result so that future retries can reuse it."""
        key = self._idempotency_key(tool_name, arguments)
        self._get_backend().set(key, result)

    def reset_idempotency_store(self) -> None:
        """Clear the idempotency store.

        Call this at the start of a fresh task execution to avoid stale
        entries from a previous unrelated task.
        """
        if self._idempotency_backend is not None:
            self._idempotency_backend.clear()

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

        # Store in idempotency store (independent of cache config).
        # Use the same sanitised name as the read path for consistency.
        if calling.arguments is not None:
            arguments: dict[str, object] = (
                calling.arguments
                if isinstance(calling.arguments, dict)
                else {}
            )
            self.set_idempotent_result(
                sanitize_tool_name(calling.tool_name), arguments, output
            )

        if self.cache and should_cache and calling.tool_name != CacheTools().name:
            input_str = ""
            if calling.arguments:
                if isinstance(calling.arguments, dict):
                    input_str = json.dumps(calling.arguments)
                else:
                    input_str = str(calling.arguments)

            self.cache.add(
                tool=sanitize_tool_name(calling.tool_name),
                input=input_str,
                output=output,
            )
