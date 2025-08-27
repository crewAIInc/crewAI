from __future__ import annotations

import asyncio
import contextvars
import functools
import hashlib
import inspect
import os
import re
import sys
import threading
import time
import weakref
from pathlib import Path
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar, ParamSpec, Concatenate, TypedDict

import portalocker
from portalocker import constants
from typing_extensions import NotRequired, Self

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")

_STATE: dict[str, int] = {"pid": os.getpid()}


def _reset_after_fork() -> None:
    """Reset in-process state in the child after a fork.

    Resets all locks and thread-local storage after a process fork
    to prevent lock contamination across processes.
    """
    global _sync_rlocks, _async_locks_by_loop, _tls, _task_depths_var, _STATE
    _sync_rlocks = {}
    _async_locks_by_loop = weakref.WeakKeyDictionary()
    _tls = threading.local()
    # Reset task-local depths for async
    _task_depths_var = contextvars.ContextVar("locked_task_depths", default=None)
    _STATE["pid"] = os.getpid()


def _ensure_same_process() -> None:
    """Ensure we're in the same process, reset if forked.

    Checks if the current PID matches the stored PID and resets
    state if a fork has occurred.
    """
    if _STATE["pid"] != os.getpid():
        _reset_after_fork()


# Automatically reset in a forked child on POSIX
_register_at_fork = getattr(os, "register_at_fork", None)
if _register_at_fork is not None:
    _register_at_fork(after_in_child=_reset_after_fork)


class LockConfig(TypedDict):
    """Configuration for portalocker locks.

    Attributes:
        mode: File open mode.
        timeout: Optional lock timeout.
        check_interval: Optional check interval.
        fail_when_locked: Whether to fail if already locked.
        flags: Portalocker lock flags.
    """

    mode: str
    timeout: NotRequired[float]
    check_interval: NotRequired[float]
    fail_when_locked: bool
    flags: portalocker.LockFlags


def _get_platform_lock_flags() -> portalocker.LockFlags:
    """Get platform-appropriate lock flags.

    Returns:
        LockFlags.EXCLUSIVE for exclusive file locking.
    """
    # Use EXCLUSIVE flag only - let portalocker handle blocking/non-blocking internally
    return constants.LockFlags.EXCLUSIVE


def _get_lock_config() -> LockConfig:
    """Get lock configuration appropriate for the platform.

    Returns:
        LockConfig dict with mode, flags, and fail_when_locked settings.
    """
    config: LockConfig = {
        "mode": "a+",
        "fail_when_locked": False,
        "flags": _get_platform_lock_flags(),
    }
    return config


LOCK_CONFIG: LockConfig = _get_lock_config()
LOCK_STALE_SECONDS = 120


def _default_lock_dir() -> Path:
    """Get or create the default lock directory.

    Returns:
        Path to ~/.crewai/locks directory, created if necessary.
    """
    lock_dir = Path.home() / ".crewai" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    # Best-effort: restrict perms on POSIX
    try:
        if os.name == "posix":
            lock_dir.chmod(0o700)
    except Exception:
        pass

    # Clean up old lock files
    _cleanup_stale_locks(lock_dir)
    return lock_dir


def _cleanup_stale_locks(lock_dir: Path, max_age_seconds: int = 86400) -> None:
    """Remove lock files older than max_age_seconds.

    Args:
        lock_dir: Directory containing lock files.
        max_age_seconds: Maximum age before considering a lock stale (default 24 hours).
    """
    try:
        current_time = time.time()
        for lock_file in lock_dir.glob("*.lock"):
            try:
                # Check if file is old and not currently locked
                file_age = current_time - lock_file.stat().st_mtime
                if file_age > max_age_seconds:
                    # Try to acquire exclusive lock - if successful, file is not in use
                    try:
                        with portalocker.Lock(
                            str(lock_file),
                            mode="a+",
                            timeout=0.01,  # Very short timeout
                            fail_when_locked=True,
                            flags=constants.LockFlags.EXCLUSIVE,
                        ):
                            pass  # We got the lock, file is not in use
                        # Safe to remove
                        lock_file.unlink(missing_ok=True)
                    except (portalocker.LockException, OSError):
                        # File is locked or can't be accessed, skip it
                        pass
            except (OSError, IOError):
                # Skip files we can't stat or process
                pass
    except Exception:
        # Cleanup is best-effort, don't fail on errors
        pass


def _hash_str(value: str) -> str:
    """Generate a short hash of a string.

    Args:
        value: String to hash.

    Returns:
        First 10 characters of SHA256 hash.
    """
    return hashlib.sha256(value.encode()).hexdigest()[:10]


def _qualname_for(func: Callable[..., Any], owner: object | None = None) -> str:
    """Get qualified name for a function.

    Args:
        func: Function to get qualified name for.
        owner: Optional owner object for the function.

    Returns:
        Fully qualified name including module and class.
    """
    target = inspect.unwrap(func)

    if inspect.ismethod(func) and getattr(func, "__self__", None) is not None:
        owner_obj = func.__self__
        cls = owner_obj if inspect.isclass(owner_obj) else owner_obj.__class__
        return f"{target.__module__}.{cls.__qualname__}.{getattr(target, '__name__', '<?>')}"

    if owner is not None:
        cls = owner if inspect.isclass(owner) else owner.__class__
        return f"{target.__module__}.{cls.__qualname__}.{getattr(target, '__name__', '<?>')}"

    qn = getattr(target, "__qualname__", None)
    if qn is not None:
        return f"{getattr(target, '__module__', target.__class__.__module__)}.{qn}"

    if isinstance(target, functools.partial):
        f = inspect.unwrap(target.func)
        return f"{getattr(f, '__module__', 'builtins')}.{getattr(f, '__qualname__', getattr(f, '__name__', '<?>'))}"

    cls = target.__class__
    return f"{cls.__module__}.{cls.__qualname__}.__call__"


def _get_lock_context(
    instance: Any | None,
    func: Callable[..., Any],
    kwargs: dict[str, Any],
) -> tuple[Path, str | None]:
    """Extract lock context from function call.

    Args:
        instance: Instance the function is called on.
        func: Function being called.
        kwargs: Keyword arguments passed to function.

    Returns:
        Tuple of (lock_file_path, collection_name).
    """
    collection_name = (
        str(kwargs.get("collection_name")) if "collection_name" in kwargs else None
    )
    lock_dir = _default_lock_dir()
    base = _qualname_for(func, owner=instance)
    safe_base = re.sub(r"[^\w.\-]+", "_", base)
    suffix = f"_{_hash_str(collection_name)}" if collection_name else ""
    path = lock_dir / f"{safe_base}{suffix}.lock"
    return path, collection_name


def _write_lock_metadata(lock_file_path: Path) -> None:
    """Write metadata to lock file for staleness detection.

    Args:
        lock_file_path: Path to the lock file.
    """
    try:
        with open(lock_file_path, "w") as f:
            f.write(f"{os.getpid()}\n{time.time()}\n")
            f.flush()
            os.fsync(f.fileno())  # Ensure data is written to disk

        # Set restrictive permissions on lock file (Unix only)
        if sys.platform not in ("win32", "cygwin"):
            try:
                lock_file_path.chmod(0o600)
            except Exception:
                pass
    except Exception:
        # Best effort - don't fail if we can't write metadata
        pass


def _check_lock_staleness(lock_file_path: Path) -> bool:
    """Check if a lock file is stale.

    Args:
        lock_file_path: Path to the lock file.

    Returns:
        True if lock is stale, False otherwise.
    """
    try:
        if not lock_file_path.exists():
            return False

        with open(lock_file_path) as f:
            lines = f.readlines()
        if len(lines) < 2:
            return True  # unreadable metadata

        pid = int(lines[0].strip())
        ts = float(lines[1].strip())

        # If the process is alive, do NOT treat as stale based on time alone.
        if sys.platform not in ("win32", "cygwin"):
            try:
                os.kill(pid, 0)
                return False  # alive → not stale
            except (OSError, ProcessLookupError):
                pass  # dead process → proceed to time check

        # Process dead: time window can be small; consider stale now
        return (time.time() - ts) > 1.0  # essentially “dead means stale”

    except Exception:
        return True


_sync_rlocks: dict[Path, threading.RLock] = {}
_sync_rlocks_guard = threading.Lock()
_tls = threading.local()


def _get_sync_rlock(path: Path) -> threading.RLock:
    """Get or create a reentrant lock for a path.

    Args:
        path: Path to get lock for.

    Returns:
        Threading RLock for the given path.
    """
    with _sync_rlocks_guard:
        lk = _sync_rlocks.get(path)
        if lk is None:
            lk = threading.RLock()
            _sync_rlocks[path] = lk
        return lk


class _SyncDepthManager:
    """Context manager for sync depth tracking.

    Tracks reentrancy depth for synchronous locks to determine
    when to acquire/release file locks.
    """

    def __init__(self, path: Path):
        """Initialize depth manager.

        Args:
            path: Path to track depth for.
        """
        self.path = path
        self.depth = 0

    def __enter__(self) -> int:
        """Enter context and increment depth.

        Returns:
            Current depth after increment.
        """
        depths = getattr(_tls, "depths", None)
        if depths is None:
            depths = {}
            _tls.depths = depths
        self.depth = depths.get(self.path, 0) + 1
        depths[self.path] = self.depth
        return self.depth

    def __exit__(self, *args: Any) -> None:
        """Exit context and decrement depth.

        Args:
            *args: Exception information if any.
        """
        depths = getattr(_tls, "depths", {})
        v = depths.get(self.path, 1) - 1
        if v <= 0:
            depths.pop(self.path, None)
        else:
            depths[self.path] = v


def _safe_to_delete(path: Path) -> bool:
    """Check if a lock file can be safely deleted.

    Args:
        path: Path to the lock file.

    Returns:
        True if file can be deleted safely, False otherwise.
    """
    try:
        with portalocker.Lock(
            str(path),
            mode="a+",
            timeout=0.01,  # very short, non-blocking-ish
            fail_when_locked=True,  # fail if someone holds it
            flags=constants.LockFlags.EXCLUSIVE,
        ):
            return True
    except Exception:
        return False


def with_lock(func: Callable[Concatenate[T, P], R]) -> Callable[Concatenate[T, P], R]:
    """Decorator for file-based cross-process locking.

    Args:
        func: Function to wrap with locking.

    Returns:
        Wrapped function with locking behavior.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        _ensure_same_process()

        path, _ = _get_lock_context(self, func, kwargs)
        local_lock = _get_sync_rlock(path)

        prune_after = False
        with local_lock:
            with _SyncDepthManager(path) as depth:
                if depth == 1:
                    # stale handling
                    if _check_lock_staleness(path) and _safe_to_delete(path):
                        try:
                            path.unlink(missing_ok=True)
                        except Exception:
                            pass

                    # acquire file lock
                    lock_config = LockConfig(
                        mode=LOCK_CONFIG["mode"],
                        fail_when_locked=LOCK_CONFIG["fail_when_locked"],
                        flags=LOCK_CONFIG["flags"],
                    )
                    with portalocker.Lock(str(path), **lock_config) as _fh:
                        _write_lock_metadata(path)
                        result = func(self, *args, **kwargs)
                    try:
                        path.unlink(missing_ok=True)
                    except Exception:
                        pass

                    prune_after = True
                else:
                    result = func(self, *args, **kwargs)

        # <-- NOW it’s safe to remove the entry
        if prune_after:
            with _sync_rlocks_guard:
                _sync_rlocks.pop(path, None)

        return result

    return wrapper


# Use weak references to avoid keeping event loops alive
_async_locks_by_loop: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[Path, asyncio.Lock]
] = weakref.WeakKeyDictionary()
_async_locks_guard = threading.Lock()
_task_depths_var: contextvars.ContextVar[dict[Path, int] | None] = (
    contextvars.ContextVar("locked_task_depths", default=None)
)


def _get_async_lock(path: Path) -> asyncio.Lock:
    """Get or create an async lock for the current event loop.

    Args:
        path: Path to get lock for.

    Returns:
        Asyncio Lock for the given path in current event loop.
    """
    loop = asyncio.get_running_loop()

    with _async_locks_guard:
        # Get locks dict for this event loop
        loop_locks = _async_locks_by_loop.get(loop)
        if loop_locks is None:
            loop_locks = {}
            _async_locks_by_loop[loop] = loop_locks

        # Get or create lock for this path
        lk = loop_locks.get(path)
        if lk is None:
            # Create lock in the context of the running loop
            lk = asyncio.Lock()
            loop_locks[path] = lk
        return lk


class _AsyncDepthManager:
    """Context manager for async task-local depth tracking.

    Tracks reentrancy depth for async locks to determine
    when to acquire/release file locks.
    """

    def __init__(self, path: Path):
        """Initialize async depth manager.

        Args:
            path: Path to track depth for.
        """
        self.path = path
        self.depths: dict[Path, int] | None = None
        self.is_reentrant = False

    def __enter__(self) -> Self:
        """Enter context and track async task depth.

        Returns:
            Self for context management.
        """
        d = _task_depths_var.get()
        if d is None:
            d = {}
            _task_depths_var.set(d)
        self.depths = d

        cur_depth = self.depths.get(self.path, 0)
        if cur_depth > 0:
            self.is_reentrant = True
            self.depths[self.path] = cur_depth + 1
        else:
            self.depths[self.path] = 1
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context and update task depth.

        Args:
            *args: Exception information if any.
        """
        if self.depths is not None:
            new_depth = self.depths.get(self.path, 1) - 1
            if new_depth <= 0:
                self.depths.pop(self.path, None)
            else:
                self.depths[self.path] = new_depth


async def _safe_to_delete_async(path: Path) -> bool:
    """Check if a lock file can be safely deleted (async).

    Args:
        path: Path to the lock file.

    Returns:
        True if file can be deleted safely, False otherwise.
    """

    def _try_lock() -> bool:
        try:
            with portalocker.Lock(
                str(path),
                mode="a+",
                timeout=0.01,  # very short, effectively non-blocking
                fail_when_locked=True,  # fail if another process holds it
                flags=constants.LockFlags.EXCLUSIVE,
            ):
                return True
        except Exception:
            return False

    return await asyncio.to_thread(_try_lock)


def async_with_lock(
    func: Callable[Concatenate[T, P], Coroutine[Any, Any, R]],
) -> Callable[Concatenate[T, P], Coroutine[Any, Any, R]]:
    """Async decorator for file-based cross-process locking.

    Args:
        func: Async function to wrap with locking.

    Returns:
        Wrapped async function with locking behavior.
    """

    @functools.wraps(func)
    async def wrapper(self: T, *args: P.args, **kwargs: P.kwargs) -> R:
        _ensure_same_process()

        path, _ = _get_lock_context(self, func, kwargs)

        with _AsyncDepthManager(path) as depth_mgr:
            if depth_mgr.is_reentrant:
                # Re-entrant within the same task: skip file lock
                return await func(self, *args, **kwargs)

            # Safer stale handling: only unlink if we can lock it first
            if _check_lock_staleness(path) and await _safe_to_delete_async(path):
                try:
                    await asyncio.to_thread(lambda: path.unlink(missing_ok=True))
                except Exception:
                    pass

            # Acquire per-loop async lock to serialize within this loop
            async_lock = _get_async_lock(path)
            await async_lock.acquire()
            try:
                # Acquire cross-process file lock in a thread
                lock_config = LockConfig(
                    mode=LOCK_CONFIG["mode"],
                    fail_when_locked=LOCK_CONFIG["fail_when_locked"],
                    flags=LOCK_CONFIG["flags"],
                )
                file_lock = portalocker.Lock(str(path), **lock_config)

                await asyncio.to_thread(file_lock.acquire)
                try:
                    # Write/refresh metadata while lock is held
                    await asyncio.to_thread(lambda: _write_lock_metadata(path))

                    result = await func(self, *args, **kwargs)
                finally:
                    # Release file lock before unlink to avoid inode race
                    try:
                        await asyncio.to_thread(file_lock.release)
                    finally:
                        # Now it's safe to unlink the path
                        try:
                            await asyncio.to_thread(
                                lambda: path.unlink(missing_ok=True)
                            )
                        except Exception:
                            pass

                return result
            finally:
                async_lock.release()

                with _async_locks_guard:
                    loop = asyncio.get_running_loop()
                    loop_locks = _async_locks_by_loop.get(loop)
                    if loop_locks is not None:
                        loop_locks.pop(path, None)

    return wrapper
