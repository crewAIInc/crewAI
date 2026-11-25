"""Tests for keeping the CLI login signed in with a refresh token."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Any, NoReturn

from crewai_core import lock_store
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
import pytest


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


def _in(seconds: float) -> int:
    return int((datetime.now() + timedelta(seconds=seconds)).timestamp())


class FakeTokenEndpoint:
    """Stands in for ``httpx.post`` and records every request it receives."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, str]]] = []
        self.responses: list[httpx.Response | Exception] = []

    def reply(self, status_code: int, body: dict[str, Any]) -> None:
        self.responses.append(httpx.Response(status_code, json=body))

    def fail(self, error: Exception) -> None:
        self.responses.append(error)

    def __call__(
        self, url: str, *, data: dict[str, str], timeout: float
    ) -> httpx.Response:
        self.requests.append((url, data))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        response.request = httpx.Request("POST", url)
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
    fake = FakeTokenEndpoint()
    monkeypatch.setattr("crewai_core.auth.oauth2.httpx.post", fake)
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


@pytest.fixture(autouse=True)
def _env(
    storage: Path,
    configured: Callable[[Oauth2Settings], None],
    endpoint: FakeTokenEndpoint,
    validator: FakeJwtValidator,
    clock: FakeClock,
) -> None:
    """Every test gets isolated storage, fixed settings, a fixed clock and no network."""


@contextmanager
def _lock_backend(backend: lock_store.LockBackend) -> Iterator[None]:
    lock_store.set_lock_backend(backend)
    try:
        yield
    finally:
        lock_store.set_lock_backend(None)


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
        self, endpoint: FakeTokenEndpoint
    ) -> None:
        _save_login()
        endpoint.reply(400, {"error": "invalid_grant"})

        with pytest.raises(AuthError, match="make sure you are logged in"):
            get_auth_token()

        assert TokenManager().get_refresh_token() is None
        with pytest.raises(AuthError, match="make sure you are logged in"):
            get_auth_token()
        assert len(endpoint.requests) == 1

    @pytest.mark.parametrize(
        "outcome",
        [
            httpx.ConnectError("offline"),
            httpx.ReadTimeout("slow"),
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

    def test_lock_that_cannot_be_taken_reads_as_logged_out(
        self, endpoint: FakeTokenEndpoint
    ) -> None:
        def unreachable(name: str, *, timeout: float) -> NoReturn:
            raise ConnectionError("Error 61 connecting to 127.0.0.1:1.")

        _save_login()

        with (
            _lock_backend(unreachable),
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


class TestConcurrentRefresh:
    @pytest.fixture
    def other_process_refreshes_first(self) -> Iterator[None]:
        """A lock backend whose acquisition finds another process already refreshed."""

        @contextmanager
        def backend(name: str, *, timeout: float) -> Iterator[None]:
            TokenManager().save_tokens(
                "access-from-other-process",
                _in(8 * 3600),
                refresh_token="refresh-2",
                token_url=TOKEN_URL,
            )
            yield

        with _lock_backend(backend):
            yield

    @pytest.mark.usefixtures("other_process_refreshes_first")
    def test_refresh_reuses_the_token_another_process_just_saved(
        self, endpoint: FakeTokenEndpoint
    ) -> None:
        _save_login()

        assert get_auth_token() == "access-from-other-process"
        assert endpoint.requests == []

    def test_logins_in_different_stores_do_not_share_a_lock(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        held: set[str] = set()

        @contextmanager
        def exclusive(name: str, *, timeout: float) -> Iterator[None]:
            if name in held:
                raise TimeoutError("lock already held")
            held.add(name)
            try:
                yield
            finally:
                held.discard(name)

        def manager_in(directory: Path) -> TokenManager:
            manager = TokenManager()
            monkeypatch.setattr(manager, "_get_secure_storage_path", lambda: directory)
            return manager

        first = manager_in(tmp_path / "a")
        same_store = manager_in(tmp_path / "a")
        other_store = manager_in(tmp_path / "b")

        with _lock_backend(exclusive), first.lock():
            with other_store.lock():
                pass
            with pytest.raises(TimeoutError, match="lock already held"):
                with same_store.lock():
                    pass


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
