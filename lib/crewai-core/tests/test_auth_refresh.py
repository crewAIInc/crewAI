"""Tests for keeping the CLI login signed in with a refresh token."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, tzinfo
import json
import os
from pathlib import Path
import sys
from typing import Any
from urllib.parse import parse_qsl

from crewai_core import lock_store, token_manager as token_manager_module
from crewai_core.auth import token as token_module
from crewai_core.auth.oauth2 import AuthenticationCommand, Oauth2Settings
from crewai_core.auth.providers.auth0 import Auth0Provider
from crewai_core.auth.providers.base_provider import BaseProvider
from crewai_core.auth.providers.entra_id import EntraIdProvider
from crewai_core.auth.providers.keycloak import KeycloakProvider
from crewai_core.auth.providers.okta import OktaProvider
from crewai_core.auth.providers.workos import WorkosProvider
from crewai_core.auth.token import AuthError, get_auth_token
from crewai_core.token_manager import TokenManager
import httpx
import portalocker.exceptions
import pytest


_REAL_STORAGE_PATH = TokenManager.__dict__["_get_secure_storage_path"]


DOMAIN = "login.example.com"
TOKEN_URL = f"https://{DOMAIN}/oauth2/token"
CLIENT_ID = "client-123"
AUDIENCE = "audience-123"
OLD_ACCESS = "old-access"
FIRST_REFRESH = "refresh-1"


def _settings(domain: str = DOMAIN) -> Oauth2Settings:
    return Oauth2Settings(
        provider="workos", client_id=CLIENT_ID, domain=domain, audience=AUDIENCE
    )


class ShiftedDatetime(datetime):
    """``datetime`` whose ``now()`` runs ``offset`` ahead, for the token manager's clock."""

    offset = timedelta(0)

    @classmethod
    def now(cls, tz: tzinfo | None = None) -> ShiftedDatetime:
        return cls.fromtimestamp(
            datetime.now(tz).timestamp() + cls.offset.total_seconds(), tz
        )


def _in(seconds: float) -> int:
    return int((ShiftedDatetime.now() + timedelta(seconds=seconds)).timestamp())


class FakeTokenEndpoint:
    """Stands in for httpx's transport and records every request it receives."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, str]]] = []
        self.responses: list[httpx.Response | Exception] = []

    def reply(self, status_code: int, body: dict[str, Any]) -> None:
        self.responses.append(httpx.Response(status_code, json=body))

    def fail(self, error: Exception) -> None:
        self.responses.append(error)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(
            (str(request.url), dict(parse_qsl(request.read().decode())))
        )
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeJwtValidator:
    """Stands in for ``validate_jwt_token``: tokens named ``bad-*`` fail validation."""

    def __init__(self) -> None:
        self.expirations: dict[str, int] = {}
        self.calls: list[dict[str, str | float]] = []

    def __call__(
        self,
        *,
        jwt_token: str,
        jwks_url: str,
        issuer: str,
        audience: str,
        timeout: float,
    ) -> dict[str, int]:
        self.calls.append(
            {
                "jwt_token": jwt_token,
                "jwks_url": jwks_url,
                "issuer": issuer,
                "audience": audience,
                "timeout": timeout,
            }
        )
        if jwt_token.startswith("bad-"):
            raise Exception("Invalid token audience.")
        return {"exp": self.expirations.get(jwt_token, _in(8 * 3600))}


class FakeClock:
    """Stands in for the ``time`` module ``get_auth_token`` reads its clock from."""

    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now


@pytest.fixture
def storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(
        TokenManager, "_get_secure_storage_path", staticmethod(lambda: tmp_path)
    )
    return tmp_path


@pytest.fixture
def configured(monkeypatch: pytest.MonkeyPatch) -> Callable[[Oauth2Settings], None]:
    """Set the provider settings the next AuthenticationCommand reads."""

    def configure(settings: Oauth2Settings) -> None:
        monkeypatch.setattr(
            Oauth2Settings, "from_settings", classmethod(lambda cls: settings)
        )

    configure(_settings())
    return configure


@pytest.fixture
def endpoint(monkeypatch: pytest.MonkeyPatch) -> FakeTokenEndpoint:
    """Every request, through any client and proxy, reaches the fake instead of the network."""
    fake = FakeTokenEndpoint()
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", fake)
    return fake


@pytest.fixture
def validator(monkeypatch: pytest.MonkeyPatch) -> FakeJwtValidator:
    fake = FakeJwtValidator()
    monkeypatch.setattr("crewai_core.auth.oauth2.validate_jwt_token", fake)
    return fake


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    fake = FakeClock()
    monkeypatch.setattr(token_module, "time", fake)
    return fake


class WallClock:
    """Moves the token manager's clock, which decides session and access-token expiry."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._monkeypatch = monkeypatch
        monkeypatch.setattr(token_manager_module, "datetime", ShiftedDatetime)

    def advance(self, by: timedelta) -> None:
        self._monkeypatch.setattr(
            ShiftedDatetime, "offset", ShiftedDatetime.offset + by
        )


@pytest.fixture
def wall_clock(monkeypatch: pytest.MonkeyPatch) -> WallClock:
    return WallClock(monkeypatch)


@pytest.fixture
def short_lock_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(token_manager_module, "_LOCK_TIMEOUT_SECONDS", 0.05)


@pytest.fixture(autouse=True)
def _env(
    storage: Path,
    configured: Callable[[Oauth2Settings], None],
    endpoint: FakeTokenEndpoint,
    validator: FakeJwtValidator,
    clock: FakeClock,
) -> None:
    """Every test gets isolated storage, fixed settings, a fixed clock and no network."""


def _save_login(
    access_token: str = OLD_ACCESS,
    expires_at: int | None = None,
    refresh_token: str | None = FIRST_REFRESH,
    token_url: str = TOKEN_URL,
) -> None:
    TokenManager().save_tokens(
        access_token,
        _in(-60) if expires_at is None else expires_at,
        refresh_token=refresh_token,
        token_url=token_url,
    )


class TestLoginSavesRefreshToken:
    def test_device_login_saves_the_refresh_token_with_its_endpoint(
        self, endpoint: FakeTokenEndpoint
    ) -> None:
        endpoint.reply(200, {"access_token": "access-1", "refresh_token": "refresh-1"})

        AuthenticationCommand()._poll_for_token(
            {"device_code": "dc", "interval": 0, "expires_in": 60}
        )

        assert TokenManager().get_token() == "access-1"
        assert TokenManager().get_refresh_token() == ("refresh-1", TOKEN_URL)

    def test_login_without_a_refresh_token_saves_only_the_access_token(
        self, endpoint: FakeTokenEndpoint
    ) -> None:
        endpoint.reply(200, {"access_token": "access-1"})

        AuthenticationCommand()._poll_for_token(
            {"device_code": "dc", "interval": 0, "expires_in": 60}
        )

        assert TokenManager().get_token() == "access-1"
        assert TokenManager().get_refresh_token() is None

    def test_a_refresh_token_cannot_be_saved_without_its_endpoint(self) -> None:
        with pytest.raises(
            ValueError, match="token_url is required to save a refresh token"
        ):
            TokenManager().save_tokens("access", _in(60), refresh_token="refresh")


class TestSilentRefresh:
    def test_expired_access_token_is_renewed_with_the_refresh_token(
        self, endpoint: FakeTokenEndpoint, validator: FakeJwtValidator
    ) -> None:
        _save_login()
        endpoint.reply(
            200, {"access_token": "new-access", "refresh_token": "refresh-2"}
        )

        assert get_auth_token() == "new-access"

        assert endpoint.requests == [
            (
                TOKEN_URL,
                {
                    "grant_type": "refresh_token",
                    "refresh_token": "refresh-1",
                    "client_id": CLIENT_ID,
                },
            )
        ]
        assert validator.calls == [
            {
                "jwt_token": "new-access",
                "jwks_url": f"https://{DOMAIN}/oauth2/jwks",
                "issuer": f"https://{DOMAIN}",
                "audience": AUDIENCE,
                "timeout": 5,
            }
        ]

    def test_rotated_refresh_token_is_saved_for_the_next_refresh(
        self, endpoint: FakeTokenEndpoint
    ) -> None:
        _save_login()
        endpoint.reply(
            200, {"access_token": "new-access", "refresh_token": "refresh-2"}
        )

        get_auth_token()

        assert TokenManager().get_token() == "new-access"
        assert TokenManager().get_refresh_token() == ("refresh-2", TOKEN_URL)

    def test_refresh_response_without_a_refresh_token_keeps_the_current_one(
        self, endpoint: FakeTokenEndpoint
    ) -> None:
        _save_login()
        endpoint.reply(200, {"access_token": "new-access"})

        assert get_auth_token() == "new-access"
        assert TokenManager().get_refresh_token() == ("refresh-1", TOKEN_URL)

    def test_valid_access_token_is_returned_without_a_network_call(
        self, endpoint: FakeTokenEndpoint
    ) -> None:
        _save_login(expires_at=_in(3600))

        assert get_auth_token() == "old-access"
        assert endpoint.requests == []

    def test_access_token_about_to_expire_is_renewed_early(
        self, endpoint: FakeTokenEndpoint
    ) -> None:
        _save_login(expires_at=_in(30))
        endpoint.reply(200, {"access_token": "new-access"})

        assert get_auth_token() == "new-access"

    def test_renewed_token_is_reused_by_later_calls(
        self, endpoint: FakeTokenEndpoint
    ) -> None:
        _save_login()
        endpoint.reply(200, {"access_token": "new-access"})

        assert get_auth_token() == "new-access"
        assert get_auth_token() == "new-access"
        assert len(endpoint.requests) == 1


class TestRefreshFailure:
    def test_rejected_refresh_token_logs_the_user_out(
        self, endpoint: FakeTokenEndpoint, caplog: pytest.LogCaptureFixture
    ) -> None:
        _save_login()
        endpoint.reply(400, {"error": "invalid_grant"})

        with (
            caplog.at_level("WARNING", logger="crewai_core.auth.oauth2"),
            pytest.raises(AuthError, match="make sure you are logged in"),
        ):
            get_auth_token()

        assert TokenManager().get_refresh_token() is None
        assert "rejected (invalid_grant)" in caplog.text
        assert "crewai login" in caplog.text
        with pytest.raises(AuthError, match="make sure you are logged in"):
            get_auth_token()
        assert len(endpoint.requests) == 1

    @pytest.mark.parametrize(
        ("reply", "logged"),
        [
            (
                httpx.Response(400, json={"error": "invalid_client"}),
                "HTTP 400 (invalid_client)",
            ),
            (
                httpx.Response(400, json={"error": "invalid_request"}),
                "HTTP 400 (invalid_request)",
            ),
            (
                httpx.Response(400, json={"error": "unsupported_grant_type"}),
                "HTTP 400 (unsupported_grant_type)",
            ),
            (
                httpx.Response(401, json={"error": "invalid_client"}),
                "HTTP 401 (invalid_client)",
            ),
            (
                httpx.Response(400, text="<html>Bad Request</html>"),
                "HTTP 400 (no recognized error code)",
            ),
            (
                httpx.Response(400, json=["invalid_grant"]),
                "HTTP 400 (no recognized error code)",
            ),
            (
                httpx.Response(400, json={"error": "refresh-1 is not valid"}),
                "HTTP 400 (no recognized error code)",
            ),
            (
                httpx.Response(400, json={"error": "refresh-1"}),
                "HTTP 400 (no recognized error code)",
            ),
        ],
        ids=[
            "invalid_client",
            "invalid_request",
            "unsupported_grant_type",
            "401",
            "non-json",
            "not-an-object",
            "unsafe-error-code",
            "token-shaped-error-code",
        ],
    )
    def test_a_rejection_other_than_invalid_grant_keeps_the_login(
        self,
        endpoint: FakeTokenEndpoint,
        clock: FakeClock,
        caplog: pytest.LogCaptureFixture,
        reply: httpx.Response,
        logged: str,
    ) -> None:
        _save_login()
        endpoint.responses.append(reply)

        with (
            caplog.at_level("WARNING", logger="crewai_core.auth.oauth2"),
            pytest.raises(AuthError, match="make sure you are logged in"),
        ):
            get_auth_token()

        assert TokenManager().get_refresh_token() == ("refresh-1", TOKEN_URL)
        assert logged in caplog.text
        assert "The login is kept." in caplog.text
        assert "refresh-1" not in caplog.text
        clock.now += 301
        endpoint.reply(200, {"access_token": "new-access"})
        assert get_auth_token() == "new-access"

    @pytest.mark.parametrize(
        "outcome",
        [
            httpx.ConnectError("offline"),
            httpx.ReadTimeout("slow"),
            httpx.Response(500, text="Internal Server Error"),
            httpx.Response(503, json={"error": "temporarily_unavailable"}),
        ],
    )
    def test_network_or_server_error_keeps_the_refresh_token_for_later(
        self,
        endpoint: FakeTokenEndpoint,
        clock: FakeClock,
        outcome: httpx.Response | Exception,
    ) -> None:
        _save_login()
        endpoint.responses.append(outcome)

        with pytest.raises(AuthError, match="make sure you are logged in"):
            get_auth_token()

        assert TokenManager().get_refresh_token() == ("refresh-1", TOKEN_URL)
        clock.now += 301
        endpoint.reply(200, {"access_token": "new-access"})
        assert get_auth_token() == "new-access"

    def test_failed_refresh_inside_the_margin_still_returns_the_current_token(
        self, endpoint: FakeTokenEndpoint
    ) -> None:
        _save_login(expires_at=_in(30))
        endpoint.fail(httpx.ConnectError("offline"))

        assert get_auth_token() == "old-access"

    def test_rotated_refresh_token_is_kept_when_its_access_token_fails_validation(
        self, endpoint: FakeTokenEndpoint, clock: FakeClock
    ) -> None:
        _save_login()
        endpoint.reply(
            200, {"access_token": "bad-access", "refresh_token": "refresh-2"}
        )

        with pytest.raises(AuthError, match="make sure you are logged in"):
            get_auth_token()
        assert TokenManager().get_token() is None

        clock.now += 301
        endpoint.reply(
            200, {"access_token": "new-access", "refresh_token": "refresh-3"}
        )
        assert get_auth_token() == "new-access"
        # The provider spent refresh-1 on the first exchange; the retry sends refresh-2.
        assert [data["refresh_token"] for _, data in endpoint.requests] == [
            "refresh-1",
            "refresh-2",
        ]

    def test_unrotated_refresh_token_survives_a_failed_validation(
        self, endpoint: FakeTokenEndpoint
    ) -> None:
        _save_login(expires_at=_in(30))
        endpoint.reply(200, {"access_token": "bad-access"})

        assert get_auth_token() == "old-access"
        assert TokenManager().get_refresh_token() == ("refresh-1", TOKEN_URL)

    @pytest.mark.parametrize(
        "reply",
        [
            httpx.Response(200, text="<html>Sign in to the Wi-Fi</html>"),
            httpx.Response(200, json=["access_token"]),
            httpx.Response(200, json={"refresh_token": "refresh-2"}),
        ],
        ids=["html", "not-an-object", "no-access-token"],
    )
    def test_unreadable_refresh_reply_reads_as_logged_out(
        self, endpoint: FakeTokenEndpoint, reply: httpx.Response
    ) -> None:
        _save_login()
        endpoint.responses.append(reply)

        with pytest.raises(AuthError, match="make sure you are logged in"):
            get_auth_token()

    @pytest.mark.usefixtures("short_lock_timeout")
    def test_lock_that_cannot_be_taken_reads_as_logged_out(
        self, endpoint: FakeTokenEndpoint
    ) -> None:
        _save_login()

        with (
            TokenManager().lock(),
            pytest.raises(AuthError, match="make sure you are logged in"),
        ):
            get_auth_token()
        assert endpoint.requests == []
        assert TokenManager().get_refresh_token() == ("refresh-1", TOKEN_URL)

    def test_settings_that_no_longer_load_read_as_logged_out(
        self, monkeypatch: pytest.MonkeyPatch, endpoint: FakeTokenEndpoint
    ) -> None:
        _save_login()
        monkeypatch.setattr(
            Oauth2Settings,
            "from_settings",
            classmethod(
                lambda cls: Oauth2Settings(
                    provider="no_such_provider", client_id="c", domain="d"
                )
            ),
        )

        with pytest.raises(AuthError, match="make sure you are logged in"):
            get_auth_token()
        assert endpoint.requests == []


class TestRetryAfterFailure:
    def test_failed_refresh_is_not_retried_by_later_calls_in_the_same_run(
        self, endpoint: FakeTokenEndpoint
    ) -> None:
        _save_login()
        endpoint.fail(httpx.ConnectTimeout("dropped"))

        for _ in range(3):
            with pytest.raises(AuthError, match="make sure you are logged in"):
                get_auth_token()

        assert len(endpoint.requests) == 1

    def test_refresh_is_retried_once_the_wait_is_over(
        self, endpoint: FakeTokenEndpoint, clock: FakeClock
    ) -> None:
        _save_login()
        endpoint.fail(httpx.ConnectTimeout("dropped"))
        with pytest.raises(AuthError, match="make sure you are logged in"):
            get_auth_token()

        clock.now += 299
        with pytest.raises(AuthError, match="make sure you are logged in"):
            get_auth_token()
        clock.now += 2
        endpoint.reply(200, {"access_token": "new-access"})

        assert get_auth_token() == "new-access"
        assert len(endpoint.requests) == 2

    def test_a_failure_for_one_login_does_not_hold_back_another(
        self,
        endpoint: FakeTokenEndpoint,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _save_login()
        endpoint.fail(httpx.ConnectTimeout("dropped"))
        with pytest.raises(AuthError, match="make sure you are logged in"):
            get_auth_token()

        other = tmp_path / "other-user"
        other.mkdir()
        monkeypatch.setattr(
            TokenManager, "_get_secure_storage_path", staticmethod(lambda: other)
        )
        _save_login()
        endpoint.reply(200, {"access_token": "new-access"})

        assert get_auth_token() == "new-access"


class TestNoRefresh:
    def test_expired_login_is_not_renewed_when_refresh_is_off(
        self, endpoint: FakeTokenEndpoint
    ) -> None:
        _save_login()

        with pytest.raises(AuthError, match="make sure you are logged in"):
            get_auth_token(refresh=False)
        assert endpoint.requests == []

    def test_token_inside_the_margin_is_returned_when_refresh_is_off(
        self, endpoint: FakeTokenEndpoint
    ) -> None:
        _save_login(expires_at=_in(30))

        assert get_auth_token(refresh=False) == "old-access"
        assert endpoint.requests == []


class TestLegacyTokens:
    def _write_legacy_file(self, storage: Path, expires_at: int) -> None:
        manager = TokenManager()
        data = {
            "access_token": "legacy-access",
            "expiration": datetime.fromtimestamp(expires_at).isoformat(),
        }
        (storage / "tokens.enc").write_bytes(
            manager.fernet.encrypt(json.dumps(data).encode())
        )

    def test_valid_legacy_token_still_works(self, storage: Path) -> None:
        self._write_legacy_file(storage, _in(3600))

        assert get_auth_token() == "legacy-access"
        assert TokenManager().get_refresh_token() is None

    def test_expired_legacy_token_asks_for_login_without_a_network_call(
        self, storage: Path, endpoint: FakeTokenEndpoint
    ) -> None:
        self._write_legacy_file(storage, _in(-60))

        with pytest.raises(AuthError, match="make sure you are logged in"):
            get_auth_token()
        assert endpoint.requests == []

    def test_no_login_at_all_raises_auth_error(self) -> None:
        with pytest.raises(AuthError, match="make sure you are logged in"):
            get_auth_token()


class TestRefreshTokenDestination:
    def test_refresh_token_is_not_sent_when_the_configured_endpoint_changed(
        self,
        configured: Callable[[Oauth2Settings], None],
        endpoint: FakeTokenEndpoint,
    ) -> None:
        _save_login()
        configured(_settings(domain="attacker.example.com"))

        with pytest.raises(AuthError, match="make sure you are logged in"):
            get_auth_token()

        assert endpoint.requests == []

    def test_refresh_token_is_not_in_the_token_file_in_plain_text(
        self, storage: Path
    ) -> None:
        _save_login(refresh_token="very-secret-refresh-token")

        assert b"very-secret-refresh-token" not in (storage / "tokens.enc").read_bytes()


@pytest.fixture
def other_process_refreshes_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """Taking the lock finds that another process refreshed while this one waited."""

    @contextmanager
    def lock(manager: TokenManager) -> Iterator[None]:
        manager.save_tokens(
            "access-from-other-process",
            _in(8 * 3600),
            refresh_token="refresh-2",
            token_url=TOKEN_URL,
            keep_login_time=True,
        )
        yield

    monkeypatch.setattr(TokenManager, "lock", lock)


class TestConcurrentRefresh:
    @pytest.mark.usefixtures("other_process_refreshes_first")
    def test_refresh_reuses_the_token_another_process_just_saved(
        self, endpoint: FakeTokenEndpoint
    ) -> None:
        _save_login()

        assert get_auth_token() == "access-from-other-process"
        assert endpoint.requests == []

    @pytest.mark.usefixtures("short_lock_timeout")
    def test_logins_in_different_stores_do_not_share_a_lock(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def manager_in(directory: Path) -> TokenManager:
            directory.mkdir(exist_ok=True)
            manager = TokenManager()
            monkeypatch.setattr(manager, "_get_secure_storage_path", lambda: directory)
            return manager

        first = manager_in(tmp_path / "a")
        same_store = manager_in(tmp_path / "a")
        other_store = manager_in(tmp_path / "b")

        with first.lock():
            with other_store.lock():
                pass
            with (
                pytest.raises(portalocker.exceptions.LockException),
                same_store.lock(),
            ):
                pass

    def test_the_lock_file_sits_next_to_the_token_file(self, storage: Path) -> None:
        with TokenManager().lock():
            assert (storage / "tokens.enc.lock").is_file()

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes and owners")
    def test_the_lock_is_in_this_users_private_credentials_directory(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr(
            TokenManager, "_get_secure_storage_path", _REAL_STORAGE_PATH
        )

        with TokenManager().lock():
            lock_files = list(tmp_path.rglob("tokens.enc.lock"))
            assert len(lock_files) == 1
            directory = lock_files[0].parent.stat()
            assert directory.st_mode & 0o777 == 0o700
            assert directory.st_uid == os.getuid()

    def test_a_lock_held_at_the_old_shared_temp_path_does_not_block_a_refresh(
        self, endpoint: FakeTokenEndpoint
    ) -> None:
        """Another local account could create or hold that file; it no longer matters."""
        _save_login()
        endpoint.reply(200, {"access_token": "new-access"})

        with lock_store.lock(f"file:{TokenManager().token_path}", timeout=1):
            assert get_auth_token() == "new-access"


class TestSessionLimits:
    """A login lasts 72 hours from its last use and ends 30 days after `crewai login`."""

    def _token_file(self, storage: Path) -> bytes:
        return (storage / "tokens.enc").read_bytes()

    def test_each_use_extends_the_login_by_another_72_hours(
        self, endpoint: FakeTokenEndpoint, wall_clock: WallClock
    ) -> None:
        _save_login()

        for day in range(1, 4):
            wall_clock.advance(timedelta(hours=71))
            endpoint.reply(200, {"access_token": f"access-{day}"})
            assert get_auth_token() == f"access-{day}"

        assert len(endpoint.requests) == 3

    def test_a_use_with_a_still_valid_access_token_also_extends_the_login(
        self, endpoint: FakeTokenEndpoint, wall_clock: WallClock
    ) -> None:
        _save_login(expires_at=_in(200 * 3600))

        wall_clock.advance(timedelta(hours=71))
        assert get_auth_token() == "old-access"
        wall_clock.advance(timedelta(hours=71))

        assert get_auth_token() == "old-access"
        assert endpoint.requests == []

    def test_a_login_idle_for_over_72_hours_is_logged_out_without_a_refresh(
        self, endpoint: FakeTokenEndpoint, wall_clock: WallClock
    ) -> None:
        _save_login()

        wall_clock.advance(timedelta(hours=72, minutes=1))

        with pytest.raises(AuthError, match="make sure you are logged in"):
            get_auth_token()
        assert endpoint.requests == []
        assert TokenManager().get_refresh_token() is None

    def test_an_idle_login_is_not_used_even_with_an_unexpired_access_token(
        self, wall_clock: WallClock
    ) -> None:
        _save_login(expires_at=_in(200 * 3600))

        wall_clock.advance(timedelta(hours=72, minutes=1))

        for refresh in (True, False):
            with pytest.raises(AuthError, match="make sure you are logged in"):
                get_auth_token(refresh=refresh)

    def test_the_login_ends_30_days_after_crewai_login_however_often_it_is_used(
        self, endpoint: FakeTokenEndpoint, wall_clock: WallClock
    ) -> None:
        _save_login()
        for _ in range(14):
            wall_clock.advance(timedelta(hours=48))
            endpoint.reply(200, {"access_token": "renewed", "refresh_token": "next"})
            assert get_auth_token() == "renewed"
        requests_sent = len(endpoint.requests)

        wall_clock.advance(timedelta(days=2, minutes=1))

        with pytest.raises(AuthError, match="make sure you are logged in"):
            get_auth_token()
        assert len(endpoint.requests) == requests_sent

    def test_an_unexpired_access_token_is_not_used_past_the_30_days(
        self, endpoint: FakeTokenEndpoint, wall_clock: WallClock
    ) -> None:
        _save_login(expires_at=_in(40 * 24 * 3600))
        for _ in range(15):
            wall_clock.advance(timedelta(hours=47, minutes=56))
            assert get_auth_token() == "old-access"

        wall_clock.advance(timedelta(hours=1, minutes=1))

        assert TokenManager().get_token() is None
        for refresh in (True, False):
            with pytest.raises(AuthError, match="make sure you are logged in"):
                get_auth_token(refresh=refresh)
        assert endpoint.requests == []

    def test_logging_in_again_starts_a_new_30_days(
        self, endpoint: FakeTokenEndpoint, wall_clock: WallClock
    ) -> None:
        _save_login()
        wall_clock.advance(timedelta(days=31))
        endpoint.reply(200, {"access_token": "access-1", "refresh_token": "refresh-9"})

        AuthenticationCommand()._poll_for_token(
            {"device_code": "dc", "interval": 0, "expires_in": 60}
        )
        wall_clock.advance(timedelta(hours=1))

        assert get_auth_token() == "access-1"

    def test_reading_the_login_without_refresh_does_not_extend_it_or_write_the_file(
        self, storage: Path, wall_clock: WallClock
    ) -> None:
        """`import crewai` reads it this way (TraceBatchManager), on every import."""
        _save_login(expires_at=_in(200 * 3600))
        saved = self._token_file(storage)

        wall_clock.advance(timedelta(hours=71))
        assert get_auth_token(refresh=False) == "old-access"
        assert self._token_file(storage) == saved
        wall_clock.advance(timedelta(hours=2))

        with pytest.raises(AuthError, match="make sure you are logged in"):
            get_auth_token(refresh=False)

    def test_a_login_saved_without_session_times_starts_counting_at_its_first_use(
        self, storage: Path, endpoint: FakeTokenEndpoint, wall_clock: WallClock
    ) -> None:
        manager = TokenManager()
        data = {
            "access_token": "legacy-access",
            "expiration": datetime.fromtimestamp(_in(-60)).isoformat(),
            "refresh_token": FIRST_REFRESH,
            "token_url": TOKEN_URL,
        }
        (storage / "tokens.enc").write_bytes(
            manager.fernet.encrypt(json.dumps(data).encode())
        )
        wall_clock.advance(timedelta(days=90))
        endpoint.reply(200, {"access_token": "new-access"})

        assert get_auth_token() == "new-access"
        wall_clock.advance(timedelta(hours=72, minutes=1))
        with pytest.raises(AuthError, match="make sure you are logged in"):
            get_auth_token()
        assert len(endpoint.requests) == 1

    @pytest.mark.usefixtures("short_lock_timeout")
    def test_a_use_while_another_process_holds_the_lock_still_returns_the_token(
        self, wall_clock: WallClock
    ) -> None:
        _save_login(expires_at=_in(200 * 3600))
        wall_clock.advance(timedelta(hours=1))

        with TokenManager().lock():
            assert get_auth_token() == "old-access"

    @pytest.mark.usefixtures("other_process_refreshes_first")
    def test_recording_a_use_keeps_a_refresh_token_another_process_rotated(
        self, wall_clock: WallClock
    ) -> None:
        _save_login(expires_at=_in(200 * 3600))
        wall_clock.advance(timedelta(hours=1))

        get_auth_token()

        assert TokenManager().get_refresh_token() == ("refresh-2", TOKEN_URL)


class TestLogout:
    def test_clearing_tokens_removes_the_refresh_token(
        self, endpoint: FakeTokenEndpoint
    ) -> None:
        _save_login()

        TokenManager().clear_tokens()

        assert TokenManager().get_refresh_token() is None
        with pytest.raises(AuthError, match="make sure you are logged in"):
            get_auth_token()
        assert endpoint.requests == []


class TestProviderScopes:
    def test_workos_requests_offline_access_for_a_refresh_token(self) -> None:
        assert WorkosProvider(_settings()).get_oauth_scopes() == [
            "openid",
            "profile",
            "email",
            "offline_access",
        ]

    def test_workos_device_code_request_asks_for_offline_access(
        self, endpoint: FakeTokenEndpoint
    ) -> None:
        endpoint.reply(200, {"device_code": "dc"})

        AuthenticationCommand()._get_device_code()

        url, data = endpoint.requests[0]
        assert url == f"https://{DOMAIN}/oauth2/device_authorization"
        assert data["scope"] == "openid profile email offline_access"

    @pytest.mark.parametrize(
        "provider",
        [
            Auth0Provider(
                Oauth2Settings(
                    provider="auth0", client_id="c", domain="d", audience="a"
                )
            ),
            OktaProvider(
                Oauth2Settings(provider="okta", client_id="c", domain="d", audience="a")
            ),
            KeycloakProvider(
                Oauth2Settings(
                    provider="keycloak",
                    client_id="c",
                    domain="d",
                    extra={"realm": "r"},
                )
            ),
        ],
        ids=["auth0", "okta", "keycloak"],
    )
    def test_other_providers_keep_their_scopes(self, provider: BaseProvider) -> None:
        assert provider.get_oauth_scopes() == ["openid", "profile", "email"]

    def test_entra_id_keeps_its_configured_scopes(self) -> None:
        provider = EntraIdProvider(
            Oauth2Settings(
                provider="entra_id",
                client_id="c",
                domain="tenant",
                audience="a",
                extra={"scope": "api://crewai/read"},
            )
        )

        assert provider.get_oauth_scopes() == [
            "openid",
            "profile",
            "email",
            "api://crewai/read",
        ]
