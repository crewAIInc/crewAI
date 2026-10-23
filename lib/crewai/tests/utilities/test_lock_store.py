"""Tests for lock_store.

We verify our own logic: the _redis_available guard, which portalocker
backend is selected, and that a custom backend can be plugged in. We trust
portalocker and redis-py to handle actual locking mechanics.
"""

from __future__ import annotations

from contextlib import contextmanager
import sys
import time
from unittest import mock

import portalocker.exceptions
import pytest
import redis.exceptions

import crewai_core.lock_store as lock_store
from crewai_core.lock_store import lock


@pytest.fixture(autouse=True)
def no_redis_url(monkeypatch):
    monkeypatch.setattr(lock_store, "_REDIS_URL", None)


@pytest.fixture(autouse=True)
def reset_backend():
    """Ensure a custom backend never leaks across tests."""
    lock_store.set_lock_backend(None)
    yield
    lock_store.set_lock_backend(None)


# _redis_available


def test_redis_not_available_without_url():
    assert lock_store._redis_available() is False


def test_redis_not_available_when_package_missing(monkeypatch):
    monkeypatch.setattr(lock_store, "_REDIS_URL", "redis://localhost:6379")
    monkeypatch.setitem(sys.modules, "redis", None)  # None → ImportError on import
    assert lock_store._redis_available() is False


def test_redis_available_with_url_and_package(monkeypatch):
    monkeypatch.setattr(lock_store, "_REDIS_URL", "redis://localhost:6379")
    monkeypatch.setitem(sys.modules, "redis", mock.MagicMock())
    assert lock_store._redis_available() is True


# lock strategy selection


def test_uses_file_lock_when_redis_unavailable():
    with mock.patch("portalocker.Lock") as mock_lock:
        with lock("file_test"):
            pass

    mock_lock.assert_called_once()
    assert "crewai:" in mock_lock.call_args.args[0]


@pytest.fixture
def redis_conn(monkeypatch):
    """A fake Redis connection whose ``lock()`` hands out one mock lock."""
    conn = mock.MagicMock()
    conn.lock.return_value.acquire.return_value = True
    monkeypatch.setattr(lock_store, "_redis_available", mock.Mock(return_value=True))
    monkeypatch.setattr(lock_store, "_redis_connection", mock.Mock(return_value=conn))
    return conn


def test_uses_redis_set_nx_lock_when_redis_available(redis_conn):
    with mock.patch("portalocker.RedisLock") as mock_pubsub_lock:
        with lock("redis_test", timeout=7):
            redis_conn.lock.return_value.release.assert_not_called()

    mock_pubsub_lock.assert_not_called()
    name = redis_conn.lock.call_args.args[0]
    assert name.startswith("crewai:")
    assert redis_conn.lock.call_args.kwargs["timeout"] == lock_store._LEASE_SECONDS
    acquire_kwargs = redis_conn.lock.return_value.acquire.call_args.kwargs
    assert acquire_kwargs["blocking_timeout"] == 7
    redis_conn.lock.return_value.release.assert_called_once()


def test_redis_lock_timeout_raises_lock_exception(redis_conn):
    redis_conn.lock.return_value.acquire.return_value = False

    with pytest.raises(portalocker.exceptions.LockException, match="redis_test"):
        with lock("redis_test", timeout=0):
            pytest.fail("body must not run without the lock")

    redis_conn.lock.return_value.release.assert_not_called()


def test_redis_lock_releases_when_body_raises(redis_conn):
    with pytest.raises(ValueError, match="body failed"):
        with lock("redis_test"):
            raise ValueError("body failed")

    redis_conn.lock.return_value.release.assert_called_once()


def test_redis_lock_lost_before_release_is_logged_not_raised(redis_conn, caplog):
    redis_conn.lock.return_value.release.side_effect = redis.exceptions.LockNotOwnedError(
        "expired"
    )

    with caplog.at_level("WARNING", logger=lock_store.__name__):
        with lock("redis_test"):
            pass

    assert "redis_test" in caplog.text


def test_redis_lock_renews_lease_while_held(redis_conn, monkeypatch):
    monkeypatch.setattr(lock_store, "_LEASE_SECONDS", 0.03)
    held = redis_conn.lock.return_value

    with lock("redis_test"):
        time.sleep(0.1)
    renewals_at_exit = held.reacquire.call_count
    time.sleep(0.05)

    assert renewals_at_exit >= 2
    assert held.reacquire.call_count == renewals_at_exit


def test_redis_lock_renewal_stops_when_lease_is_lost(redis_conn, monkeypatch, caplog):
    monkeypatch.setattr(lock_store, "_LEASE_SECONDS", 0.03)
    held = redis_conn.lock.return_value
    held.reacquire.side_effect = redis.exceptions.LockNotOwnedError("expired")

    with caplog.at_level("WARNING", logger=lock_store.__name__):
        with lock("redis_test"):
            time.sleep(0.1)

    assert held.reacquire.call_count == 1
    assert "redis_test" in caplog.text


# custom backend


def test_custom_backend_is_used():
    calls = []

    @contextmanager
    def fake_backend(name, *, timeout):
        calls.append((name, timeout))
        yield

    lock_store.set_lock_backend(fake_backend)

    # The default file/redis path must not be touched when overridden.
    with mock.patch("portalocker.Lock") as mock_lock:
        with lock("custom_test", timeout=5):
            pass

    mock_lock.assert_not_called()
    assert calls == [("custom_test", 5)]


def test_clearing_backend_restores_default():
    @contextmanager
    def fake_backend(name, *, timeout):
        yield

    lock_store.set_lock_backend(fake_backend)
    lock_store.set_lock_backend(None)

    with mock.patch("portalocker.Lock") as mock_lock:
        with lock("after_clear"):
            pass

    mock_lock.assert_called_once()
