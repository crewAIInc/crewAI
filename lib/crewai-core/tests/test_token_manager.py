"""Regression tests for TokenManager storage edge cases."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from crewai_core.token_manager import TokenManager


@pytest.fixture
def storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate the credentials directory per test."""
    monkeypatch.setattr(
        TokenManager,
        "_get_secure_storage_path",
        staticmethod(lambda: tmp_path),
    )
    return tmp_path


def test_round_trip(storage: Path) -> None:
    tm = TokenManager()
    tm.save_tokens("tok-abc", int(time.time()) + 3600)
    assert tm.get_token() == "tok-abc"


def test_missing_file_returns_none(storage: Path) -> None:
    assert TokenManager().get_token() is None


def test_expired_token_returns_none(storage: Path) -> None:
    tm = TokenManager()
    tm.save_tokens("tok-abc", int(time.time()) - 1)
    assert tm.get_token() is None


def test_undecryptable_token_returns_none(storage: Path) -> None:
    """A token file the current key cannot decrypt means logged out, not a crash.

    The stored ``secret.key`` can be removed by a cleanup tool or a partial
    reset while ``tokens.enc`` survives; the next ``TokenManager()`` then
    generates a fresh key and ``get_token()`` must report "no token" (which
    ``get_auth_token`` turns into the login prompt) instead of raising
    ``cryptography.fernet.InvalidToken``.
    """
    tm = TokenManager()
    tm.save_tokens("tok-abc", int(time.time()) + 3600)
    (storage / "secret.key").unlink()

    assert TokenManager().get_token() is None
