"""Tests for lock_store."""

from __future__ import annotations

import sys
import threading
import time
from unittest import mock

import portalocker.exceptions
import pytest

import crewai.utilities.lock_store as lock_store
from crewai.utilities.lock_store import lock


@pytest.fixture(autouse=True)
def no_redis_url(monkeypatch):
    """Unset REDIS_URL for every test so file-based locking is used by default."""
    monkeypatch.setattr(lock_store, "_REDIS_URL", None)


# ---------------------------------------------------------------------------
# _redis_available
# ---------------------------------------------------------------------------


def test_redis_not_available_without_url():
    assert lock_store._redis_available() is False


def test_redis_not_available_when_package_missing(monkeypatch):
    monkeypatch.setattr(lock_store, "_REDIS_URL", "redis://localhost:6379")
    # Setting a key to None in sys.modules causes ImportError on import
    monkeypatch.setitem(sys.modules, "redis", None)
    assert lock_store._redis_available() is False


def test_redis_available_with_url_and_package(monkeypatch):
    monkeypatch.setattr(lock_store, "_REDIS_URL", "redis://localhost:6379")
    monkeypatch.setitem(sys.modules, "redis", mock.MagicMock())
    assert lock_store._redis_available() is True


# ---------------------------------------------------------------------------
# file-based lock
# ---------------------------------------------------------------------------


def test_lock_yields():
    with lock("basic"):
        pass


def test_lock_releases_on_exception():
    with pytest.raises(ValueError):
        with lock("on_exception"):
            raise ValueError("boom")

    # would hang or raise LockException if the lock was not released
    with lock("on_exception"):
        pass


def test_lock_is_mutually_exclusive_across_threads():
    concurrent_holders = 0
    max_concurrent = 0
    counter_lock = threading.Lock()
    barrier = threading.Barrier(5)

    def worker():
        nonlocal concurrent_holders, max_concurrent
        barrier.wait()  # all threads compete at the same time
        with lock("mutex", timeout=10):
            with counter_lock:
                concurrent_holders += 1
                max_concurrent = max(max_concurrent, concurrent_holders)
            time.sleep(0.01)  # hold briefly so overlap is detectable if locking fails
            with counter_lock:
                concurrent_holders -= 1

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert all(not t.is_alive() for t in threads), "threads did not finish in time"

    assert max_concurrent == 1


def test_lock_timeout_raises_when_held():
    acquired = threading.Event()
    release = threading.Event()

    def holder():
        with lock("timeout_test", timeout=10):
            acquired.set()
            release.wait()

    t = threading.Thread(target=holder)
    t.start()
    acquired.wait()

    try:
        with pytest.raises(portalocker.exceptions.LockException):
            with lock("timeout_test", timeout=0.1):
                pass
    finally:
        release.set()
        t.join(timeout=10)
    assert not t.is_alive(), "holder thread did not finish in time"


def test_different_names_are_independent():
    with lock("alpha"):
        with lock("beta"):
            pass  # would deadlock if names mapped to the same lock


# ---------------------------------------------------------------------------
# Redis path
# ---------------------------------------------------------------------------


def test_redis_lock_used_when_available(monkeypatch):
    fake_conn = mock.MagicMock()
    monkeypatch.setattr(lock_store, "_REDIS_URL", "redis://localhost:6379")
    monkeypatch.setattr(lock_store, "_redis_available", mock.Mock(return_value=True))
    monkeypatch.setattr(lock_store, "_redis_connection", mock.Mock(return_value=fake_conn))

    with mock.patch("portalocker.RedisLock") as mock_redis_lock:
        with lock("redis_test"):
            pass

    mock_redis_lock.assert_called_once()
    kwargs = mock_redis_lock.call_args.kwargs
    assert kwargs["channel"].startswith("crewai:")
    assert kwargs["connection"] is fake_conn
