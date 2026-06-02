"""Tests for lock_store.

We verify our own logic: the _redis_available guard and which portalocker
backend is selected. We trust portalocker to handle actual locking mechanics.
"""

from __future__ import annotations

from contextlib import contextmanager
import sys
import types
from unittest import mock

import pytest

import crewai_core.lock_store as lock_store
from crewai_core.lock_store import lock


@pytest.fixture(autouse=True)
def no_redis_url(monkeypatch):
    monkeypatch.setattr(lock_store, "_REDIS_URL", None)


@pytest.fixture(autouse=True)
def reset_backend(monkeypatch):
    """Ensure backend overrides never leak across tests."""
    monkeypatch.setattr(lock_store, "_LOCK_FACTORY_SPEC", None)
    lock_store._env_lock_factory.cache_clear()
    lock_store.set_lock_backend(None)
    yield
    lock_store.set_lock_backend(None)
    lock_store._env_lock_factory.cache_clear()


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


# backend override


def test_override_backend_is_used():
    calls = []

    @contextmanager
    def fake_backend(name, *, timeout):
        calls.append((name, timeout))
        yield

    lock_store.set_lock_backend(fake_backend)

    # The default file/redis path must not be touched when overridden.
    with mock.patch("portalocker.Lock") as mock_lock:
        with lock("override_test", timeout=5):
            pass

    mock_lock.assert_not_called()
    assert calls == [("override_test", 5)]


def test_reset_restores_default_backend():
    @contextmanager
    def fake_backend(name, *, timeout):
        yield

    lock_store.set_lock_backend(fake_backend)
    lock_store.set_lock_backend(None)

    with mock.patch("portalocker.Lock") as mock_lock:
        with lock("after_reset"):
            pass

    mock_lock.assert_called_once()


def test_get_lock_backend_reflects_override():
    assert lock_store.get_lock_backend() is lock_store._default_lock

    @contextmanager
    def fake_backend(name, *, timeout):
        yield

    lock_store.set_lock_backend(fake_backend)
    assert lock_store.get_lock_backend() is fake_backend


# CREWAI_LOCK_FACTORY env import-path


def _install_env_factory(monkeypatch, factory, modname="fakelocks", attr="lock"):
    """Point CREWAI_LOCK_FACTORY at ``factory`` via a registered fake module."""
    module = types.ModuleType(modname)
    setattr(module, attr, factory)
    monkeypatch.setitem(sys.modules, modname, module)
    monkeypatch.setattr(lock_store, "_LOCK_FACTORY_SPEC", f"{modname}:{attr}")
    lock_store._env_lock_factory.cache_clear()


def test_env_factory_used_when_spec_set(monkeypatch):
    calls = []

    @contextmanager
    def fake_backend(name, *, timeout):
        calls.append((name, timeout))
        yield

    _install_env_factory(monkeypatch, fake_backend)

    with mock.patch("portalocker.Lock") as mock_lock:
        with lock("env_test", timeout=7):
            pass

    mock_lock.assert_not_called()
    assert calls == [("env_test", 7)]
    assert lock_store.get_lock_backend() is fake_backend


def test_programmatic_override_takes_precedence_over_env(monkeypatch):
    @contextmanager
    def env_backend(name, *, timeout):
        raise AssertionError("env backend should not be used")
        yield  # pragma: no cover

    used = []

    @contextmanager
    def code_backend(name, *, timeout):
        used.append(name)
        yield

    _install_env_factory(monkeypatch, env_backend)
    lock_store.set_lock_backend(code_backend)

    with lock("precedence_test"):
        pass

    assert used == ["precedence_test"]
    assert lock_store.get_lock_backend() is code_backend


def test_env_factory_is_cached(monkeypatch):
    @contextmanager
    def fake_backend(name, *, timeout):
        yield

    _install_env_factory(monkeypatch, fake_backend)

    with lock("a"):
        pass

    # Remove the module: a cached factory must keep working without re-importing.
    monkeypatch.delitem(sys.modules, "fakelocks")
    with lock("b"):
        pass

    assert lock_store.get_lock_backend() is fake_backend


def test_invalid_spec_raises(monkeypatch):
    monkeypatch.setattr(lock_store, "_LOCK_FACTORY_SPEC", "no_colon_here")
    lock_store._env_lock_factory.cache_clear()

    with pytest.raises(ValueError, match="module:callable"):
        with lock("bad_spec"):
            pass


def test_non_callable_factory_raises_with_context(monkeypatch):
    # Resolve the spec to a non-callable attribute.
    _install_env_factory(monkeypatch, "not a callable", attr="lock")

    with pytest.raises(TypeError, match="CREWAI_LOCK_FACTORY"):
        with lock("bad_factory"):
            pass


def test_env_factory_used_after_reset(monkeypatch):
    """Clearing the in-process override falls back to the env factory."""
    seen = []

    @contextmanager
    def env_backend(name, *, timeout):
        seen.append(name)
        yield

    @contextmanager
    def code_backend(name, *, timeout):
        raise AssertionError("override should have been cleared")
        yield  # pragma: no cover

    _install_env_factory(monkeypatch, env_backend)
    lock_store.set_lock_backend(code_backend)
    lock_store.set_lock_backend(None)

    with lock("after_reset_env"):
        pass

    assert seen == ["after_reset_env"]
    assert lock_store.get_lock_backend() is env_backend
