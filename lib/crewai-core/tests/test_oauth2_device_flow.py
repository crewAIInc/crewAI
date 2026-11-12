"""Device-flow polling tests for the OAuth2 login."""

from __future__ import annotations

from unittest.mock import MagicMock

from crewai_core.auth.oauth2 import AuthenticationCommand
import pytest


class _Provider:
    """Provider stub; the flow only asks it for endpoint URLs and identifiers."""

    def get_client_id(self) -> str:
        return "client-id"

    def get_authorize_url(self) -> str:
        return "https://example.test/device"

    def get_token_url(self) -> str:
        return "https://example.test/token"

    def get_issuer(self) -> str:
        return "https://example.test/"

    def get_jwks_url(self) -> str:
        return "https://example.test/jwks"

    def get_audience(self) -> str:
        return "crewai-api"

    def get_oauth_scopes(self) -> list[str]:
        return ["openid"]


def _command() -> AuthenticationCommand:
    """Build a command without reading the user's settings or token store."""
    command = object.__new__(AuthenticationCommand)
    command.oauth2_provider = _Provider()  # type: ignore[assignment]
    command.token_manager = MagicMock()
    return command


def test_polling_defaults_to_five_seconds_when_interval_is_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RFC 8628 makes `interval` optional; omitting it must not abort login."""
    responses = [
        {"error": "authorization_pending"},
        {"access_token": "signed-jwt"},
    ]
    status_codes = [400, 200]
    requests: list[dict[str, str]] = []

    def post(url: str, data: dict[str, str], timeout: int) -> MagicMock:
        requests.append(data)
        response = MagicMock()
        response.status_code = status_codes[len(requests) - 1]
        response.json.return_value = responses[len(requests) - 1]
        return response

    sleep = MagicMock()
    clock = MagicMock(side_effect=[0, 1, 2, 3])
    monkeypatch.setattr("crewai_core.auth.oauth2.httpx.post", post)
    monkeypatch.setattr("crewai_core.auth.oauth2.time.sleep", sleep)
    monkeypatch.setattr("crewai_core.auth.oauth2.time.monotonic", clock)
    monkeypatch.setattr(
        "crewai_core.auth.oauth2.validate_jwt_token",
        lambda **_kwargs: {"exp": 9999999999},
    )

    _command()._poll_for_token(
        {
            "device_code": "device-code",
            "user_code": "ABCD-EFGH",
            "expires_in": 600,
        }
    )

    assert len(requests) == 2
    sleep.assert_called_once_with(5)
