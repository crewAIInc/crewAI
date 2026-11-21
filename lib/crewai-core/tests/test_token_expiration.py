"""Token expiration must describe an instant, independent of local timezone."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import time

from crewai_core.token_manager import TokenManager
import pytest


@pytest.fixture
def token_manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TokenManager:
    """Use real encrypted storage without reading the user's credentials."""
    monkeypatch.setattr(
        TokenManager, "_get_secure_storage_path", staticmethod(lambda: tmp_path)
    )
    return TokenManager()


@pytest.fixture
def local_timezone(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Callable[[str], None]]:
    """Restore libc's local timezone as well as the TZ environment variable."""
    if not hasattr(time, "tzset"):
        pytest.skip("Changing the process timezone requires time.tzset")
    with monkeypatch.context() as context:

        def set_timezone(value: str) -> None:
            context.setenv("TZ", value)
            time.tzset()

        set_timezone("UTC0")
        try:
            yield set_timezone
        finally:
            context.undo()
            time.tzset()


@pytest.mark.parametrize("read_timezone", ["UTC0", "UTC-8", "UTC+8"])
@pytest.mark.parametrize("seconds_until_expiry", [-3600, 3600])
def test_token_round_trip_after_timezone_change(
    token_manager: TokenManager,
    local_timezone: Callable[[str], None],
    read_timezone: str,
    seconds_until_expiry: int,
) -> None:
    """Changing timezone must neither invalidate nor prolong a saved token."""
    token_manager.save_tokens("test-token", int(time.time()) + seconds_until_expiry)

    local_timezone(read_timezone)

    expected = "test-token" if seconds_until_expiry > 0 else None
    assert token_manager.get_token() == expected


@pytest.mark.parametrize("offset_hours", [-8, 0, 8, None])
@pytest.mark.parametrize("seconds_until_expiry", [-3600, 3600])
def test_read_aware_and_legacy_expiration(
    token_manager: TokenManager,
    offset_hours: int | None,
    seconds_until_expiry: int,
) -> None:
    """Read offset-bearing ISO dates and retain support for old naive dates."""
    tz = None if offset_hours is None else timezone(timedelta(hours=offset_hours))
    expiration = datetime.now(tz) + timedelta(seconds=seconds_until_expiry)
    payload = json.dumps(
        {"access_token": "test-token", "expiration": expiration.isoformat()}
    ).encode()
    token_manager._atomic_write_secure_file(
        "tokens.enc", token_manager.fernet.encrypt(payload)
    )

    expected = "test-token" if seconds_until_expiry > 0 else None
    assert token_manager.get_token() == expected
