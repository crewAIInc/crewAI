"""
Fix for CrewAI #5802: Tool re-execution on task retry has no idempotency guard

Root cause:
  _check_tool_repeated_usage() only checks the last used tool in memory.
  When a task fails and retries, tools_handler is reset, so last_used_tool becomes None.
  The duplicate tool call executes again — causing duplicate payments/emails/trades.

Fix:
  Add a durable idempotency layer that persists across task retries.
  Each tool call gets a stable hash (tool_name + arguments).
  If the hash exists in durable storage, return cached result instead of re-executing.

Usage:
  from crewai.tools.idempotency import IdempotencyGuard, idempotent

  @tool
  @idempotent(storage_backend="file")
  def send_payment(amount: float, recipient: str) -> str:
      result = stripe.charge(amount, recipient)
      return result
"""

from __future__ import annotations
import hashlib
import json
from typing import Any, Dict, Optional, Tuple
from pathlib import Path


class IdempotencyGuard:
    """
    Durable idempotency guard for CrewAI tools.

    Survives task retries, worker re-dispatch, and process restarts.
    Based on CCS (Conformance Checking Standard) Identity dimension.
    """

    # Class-level cache (shared within a process)
    _storage: Dict[str, Any] = {}
    # File path for durable backend (shared across all instances using file backend)
    _storage_path: Optional[Path] = None

    def __init__(self, tool_name: str, storage_backend: str = "memory"):
        """
        Args:
            tool_name: Name of the tool (used in hash computation)
            storage_backend: "memory" (default) or "file" (durable)
        """
        self.tool_name = tool_name
        self.storage_backend = storage_backend

        if storage_backend == "file":
            IdempotencyGuard._storage_path = Path(".ccs_idempotency.json")
            self._load()

    def _load(self):
        """Load storage from file (if using file backend)."""
        if (
            self.storage_backend == "file"
            and IdempotencyGuard._storage_path
            and IdempotencyGuard._storage_path.exists()
        ):
            with open(IdempotencyGuard._storage_path, "r") as f:
                IdempotencyGuard._storage = json.load(f)

    def _compute_hash(self, call_key: Dict[str, Any]) -> str:
        """Compute stable hash from tool_name + call key (args + kwargs)"""
        key = json.dumps({
            "tool": self.tool_name,
            "call_key": call_key
        }, sort_keys=True, default=str)
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def _persist(self):
        """Persist storage to disk (if using file backend)"""
        if self.storage_backend == "file" and IdempotencyGuard._storage_path:
            with open(IdempotencyGuard._storage_path, "w") as f:
                json.dump(IdempotencyGuard._storage, f)

    def is_duplicate(self, call_key: Dict[str, Any]) -> bool:
        """Check if this tool call has already been executed"""
        h = self._compute_hash(call_key)
        return h in IdempotencyGuard._storage

    def get_cached_result(self, call_key: Dict[str, Any]) -> Any:
        """Get cached result for duplicate call"""
        h = self._compute_hash(call_key)
        return IdempotencyGuard._storage.get(h)

    def record(self, call_key: Dict[str, Any], result: Any):
        """Record tool execution result"""
        h = self._compute_hash(call_key)
        IdempotencyGuard._storage[h] = result
        self._persist()

    @classmethod
    def reset(cls):
        """Reset storage (for testing)"""
        cls._storage = {}
        if cls._storage_path and cls._storage_path.exists():
            cls._storage_path.unlink()


def idempotent(storage_backend: str = "memory"):
    """
    Decorator to make a tool idempotent.

    Usage:
        @tool
        @idempotent(storage_backend="file")
        def send_payment(amount: float, recipient: str) -> str:
            stripe.charge(amount, recipient)
            return "payment sent"
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            guard = IdempotencyGuard(
                tool_name=func.__name__, storage_backend=storage_backend
            )

            # Build call key from BOTH positional and keyword arguments
            call_key = {
                "args": args,
                "kwargs": {k: v for k, v in sorted(kwargs.items())},
            }

            if guard.is_duplicate(call_key):
                cached = guard.get_cached_result(call_key)
                print(
                    f"[CCS] Blocked duplicate: {func.__name__}"
                    f", returning cached result"
                )
                return cached

            # Execute
            result = func(*args, **kwargs)
            guard.record(call_key, result)
            return result

        return wrapper

    return decorator
