"""Tests for lock_store."""

from __future__ import annotations

import threading
from unittest import mock

import portalocker.exceptions
import pytest

import crewai.utilities.lock_store as lock_store
from crewai.utilities.lock_store import lock


# ---------------------------------------------------------------------------
# _redis_available
# ---------------------------------------------------------------------------


def test_redis_available_no_url():
    with mock.patch.object(lock_store, "_REDIS_URL", None):
        assert lock_store._redis_available() is False


def test_redis_available_url_but_no_package():
    with mock.patch.object(lock_store, "_REDIS_URL", "redis://localhost:6379"):
        with mock.patch("builtins.__import__", side_effect=ImportError):
            assert lock_store._redis_available() is False


def test_redis_available_url_and_package():
    with mock.patch.object(lock_store, "_REDIS_URL", "redis://localhost:6379"):
        with mock.patch.dict("sys.modules", {"redis": mock.MagicMock()}):
            assert lock_store._redis_available() is True


# ---------------------------------------------------------------------------
# file-based lock (no Redis)
# ---------------------------------------------------------------------------


def test_lock_acquires_and_releases():
    with mock.patch.object(lock_store, "_redis_available", return_value=False):
        entered = False
        with lock("test_basic"):
            entered = True
        assert entered


def test_lock_releases_on_exception():
    with mock.patch.object(lock_store, "_redis_available", return_value=False):
        with pytest.raises(ValueError):
            with lock("test_exception"):
                raise ValueError("boom")

        # lock should be released — acquiring again must succeed
        with lock("test_exception"):
            pass


def test_lock_mutual_exclusion_across_threads():
    with mock.patch.object(lock_store, "_redis_available", return_value=False):
        results: list[int] = []
        barrier = threading.Barrier(2)

        def worker(val: int) -> None:
            barrier.wait()
            with lock("test_mutex", timeout=5):
                results.append(val)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sorted(results) == [0, 1]


def test_lock_timeout_raises():
    with mock.patch.object(lock_store, "_redis_available", return_value=False):
        acquired = threading.Event()
        hold = threading.Event()

        def holder() -> None:
            with lock("test_timeout", timeout=5):
                acquired.set()
                hold.wait()

        t = threading.Thread(target=holder)
        t.start()
        acquired.wait()

        try:
            with pytest.raises(portalocker.exceptions.LockException):
                with lock("test_timeout", timeout=0.1):
                    pass
        finally:
            hold.set()
            t.join()


def test_lock_namespaced_separately():
    """Two different lock names must not block each other."""
    with mock.patch.object(lock_store, "_redis_available", return_value=False):
        with lock("lock_a"):
            with lock("lock_b"):
                pass


# ---------------------------------------------------------------------------
# Redis path (mocked)
# ---------------------------------------------------------------------------


def test_lock_uses_redis_when_available():
    fake_redis_lock = mock.MagicMock()
    fake_redis_lock.__enter__ = mock.Mock(return_value=None)
    fake_redis_lock.__exit__ = mock.Mock(return_value=False)

    with mock.patch.object(lock_store, "_redis_available", return_value=True):
        with mock.patch.object(lock_store, "_redis_connection", return_value=mock.MagicMock()):
            with mock.patch("portalocker.RedisLock", return_value=fake_redis_lock) as mock_rl:
                with lock("test_redis"):
                    pass

                mock_rl.assert_called_once()
                _, kwargs = mock_rl.call_args
                assert kwargs["channel"].startswith("crewai:")
