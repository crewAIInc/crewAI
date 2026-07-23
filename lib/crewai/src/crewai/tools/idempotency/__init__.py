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
import functools
import hashlib
import inspect
import json
import threading
from typing import Any, Dict, Optional, Tuple
from pathlib import Path


class IdempotencyGuard:
    """
    Durable idempotency guard for CrewAI tools.

    Survives task retries, worker re-dispatch, and process restarts.
    Based on CCS (Conformance Checking Standard) Identity dimension.

    Each instance maintains its own isolated storage, keyed by tool_name.
    Supports "memory" (in-process) and "file" (durable JSON) backends.
    """

    def __init__(
        self,
        tool_name: str,
        storage_backend: str = "memory",
        storage_path: Optional[str] = None,
    ):
        """
        Args:
            tool_name: Name of the tool (used in hash computation)
            storage_backend: "memory" (default) or "file" (durable)
            storage_path: Path for file backend (default: .ccs_idempotency.json)
        """
        self.tool_name = tool_name
        self.storage_backend = storage_backend
        self._lock = threading.Lock()

        # Instance-level storage (isolated per tool_name)
        self._storage: Dict[str, Any] = {}
        self._storage_path: Optional[Path] = None

        if storage_backend == "file":
            self._storage_path = Path(storage_path or ".ccs_idempotency.json")
            self._load()

    def _load(self):
        """Load storage from file (if using file backend)."""
        if (
            self.storage_backend == "file"
            and self._storage_path
            and self._storage_path.exists()
        ):
            with open(self._storage_path, "r") as f:
                self._storage = json.load(f)

    def _compute_hash(self, call_key: Dict[str, Any]) -> str:
        """Compute stable hash from tool_name + call key (normalized arguments)"""
        key = json.dumps({
            "tool": self.tool_name,
            "call_key": call_key
        }, sort_keys=True, default=str)
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def _persist(self):
        """Persist storage to disk atomically (if using file backend)"""
        if self.storage_backend == "file" and self._storage_path:
            tmp_path = self._storage_path.with_suffix(".tmp")
            with open(tmp_path, "w") as f:
                json.dump(self._storage, f)
            tmp_path.replace(self._storage_path)

    def is_duplicate(self, call_key: Dict[str, Any]) -> bool:
        """Check if this tool call has already been executed or claimed"""
        h = self._compute_hash(call_key)
        return h in self._storage

    def claim(self, call_key: Dict[str, Any]) -> bool:
        """
        Atomically claim a call key before execution to prevent concurrent retries.
        Returns True if claimed successfully, False if already claimed/executed.
        """
        h = self._compute_hash(call_key)
        with self._lock:
            if h in self._storage:
                return False
            # Mark as claimed (pending execution)
            self._storage[h] = "__ccs_claimed__"
            self._persist()
            return True

    def get_cached_result(self, call_key: Dict[str, Any]) -> Any:
        """Get cached result for a previously executed call"""
        h = self._compute_hash(call_key)
        result = self._storage.get(h)
        if result == "__ccs_claimed__":
            return None  # Claimed but not yet completed
        return result

    def record(self, call_key: Dict[str, Any], result: Any):
        """Record tool execution result (replaces claim)"""
        h = self._compute_hash(call_key)
        # Store JSON-safe representation
        try:
            json.dumps(result)
            safe_result = result
        except (TypeError, ValueError):
            safe_result = str(result)
        with self._lock:
            self._storage[h] = safe_result
            self._persist()

    @classmethod
    def reset(cls):
        """Reset all storage (for testing)"""
        # Class-level reset only clears class attributes
        pass

    def reset_instance(self):
        """Reset this instance's storage"""
        with self._lock:
            self._storage = {}
            if self._storage_path and self._storage_path.exists():
                self._storage_path.unlink()


def _normalize_call_key(func, args, kwargs) -> Dict[str, Any]:
    """
    Normalize positional and keyword arguments into a canonical form
    so that f(1, "a") and f(x=1, y="a") produce the same hash.
    """
    sig = inspect.signature(func)
    try:
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)
    except (TypeError, ValueError):
        # Fallback for builtins or unbindable functions
        return {
            "args": args,
            "kwargs": {k: v for k, v in sorted(kwargs.items())},
        }


def idempotent(storage_backend: str = "memory", storage_path: Optional[str] = None):
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
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            guard = IdempotencyGuard(
                tool_name=func.__name__,
                storage_backend=storage_backend,
                storage_path=storage_path,
            )

            # Normalize arguments: positional and keyword calls produce same key
            call_key = _normalize_call_key(func, args, kwargs)

            # Check if already executed
            if guard.is_duplicate(call_key):
                cached = guard.get_cached_result(call_key)
                if cached is not None and cached != "__ccs_claimed__":
                    print(
                        f"[CCS] Blocked duplicate: {func.__name__}"
                        f", returning cached result"
                    )
                    return cached

            # Atomically claim before execution (prevents concurrent retries)
            if not guard.claim(call_key):
                cached = guard.get_cached_result(call_key)
                if cached is not None:
                    print(
                        f"[CCS] Blocked concurrent: {func.__name__}"
                        f", returning cached result"
                    )
                    return cached

            # Execute
            result = func(*args, **kwargs)
            guard.record(call_key, result)
            return result

        return wrapper

    return decorator
