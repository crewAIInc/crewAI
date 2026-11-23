"""Tests for lock_store.

We verify our own logic: the _redis_available guard, which portalocker
backend is selected, and that a custom backend can be plugged in. We trust
portalocker to handle actual locking mechanics.
"""

from __future__ import annotations

from contextlib import contextmanager
from hashlib import md5
import os
from pathlib import Path
import sys
import tempfile
from unittest import mock

import pytest

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
    lock_path = mock_lock.call_args.args[0]
    basename = os.path.basename(lock_path)
    expected_digest = md5(b"file_test", usedforsecurity=False).hexdigest()
    assert basename == f"crewai_{expected_digest}.lock"
    assert ":" not in basename


def test_file_lock_creates_distinct_regular_lock_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))

    with lock("portable_file_lock_a"):
        pass
    with lock("portable_file_lock_b"):
        pass

    expected_names = sorted(
        f"crewai_{md5(name.encode(), usedforsecurity=False).hexdigest()}.lock"
        for name in ("portable_file_lock_a", "portable_file_lock_b")
    )
    assert sorted(os.listdir(tmp_path)) == expected_names
    for name in expected_names:
        assert (tmp_path / name).is_file()


def test_uses_redis_lock_when_redis_available(monkeypatch):
    fake_conn = mock.MagicMock()
    monkeypatch.setattr(lock_store, "_redis_available", mock.Mock(return_value=True))
    monkeypatch.setattr(lock_store, "_redis_connection", mock.Mock(return_value=fake_conn))

    with mock.patch("portalocker.RedisLock") as mock_redis_lock:
        with lock("redis_test"):
            pass

    mock_redis_lock.assert_called_once()
    kwargs = mock_redis_lock.call_args.kwargs
    assert kwargs["channel"].startswith("crewai:")
    assert kwargs["connection"] is fake_conn


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
