"""Tracing reads the saved CLI login; an expired one now renews itself."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from crewai.events.listeners.tracing.trace_batch_manager import TraceBatchManager
from crewai.telemetry.tracing.grants import resolve_tracing_credential
from crewai_core.auth.oauth2 import Oauth2Settings
from crewai_core.token_manager import TokenManager
import httpx
import pytest


TOKEN_URL = "https://login.example.com/oauth2/token"


@pytest.fixture(autouse=True)
def expired_login(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> list[httpx.Response]:
    """A saved login whose access token expired; returns the token endpoint's replies."""
    monkeypatch.delenv("CREWAI_USER_PAT", raising=False)
    monkeypatch.setattr(
        "crewai.telemetry.tracing.grants.get_platform_integration_token", lambda: None
    )
    monkeypatch.setattr(
        TokenManager, "_get_secure_storage_path", staticmethod(lambda: tmp_path)
    )
    settings = Oauth2Settings(
        provider="workos", client_id="client", domain="login.example.com"
    )
    monkeypatch.setattr(
        Oauth2Settings, "from_settings", classmethod(lambda cls: settings)
    )
    monkeypatch.setattr(
        "crewai_core.auth.oauth2.validate_jwt_token",
        lambda **_: {"exp": int((datetime.now() + timedelta(hours=8)).timestamp())},
    )
    responses: list[httpx.Response] = []

    def post(url: str, *, data: dict[str, str], timeout: float) -> httpx.Response:
        assert url == TOKEN_URL
        return responses.pop(0)

    monkeypatch.setattr("crewai_core.auth.oauth2.httpx.post", post)
    TokenManager().save_tokens(
        "old-access",
        int((datetime.now() - timedelta(minutes=5)).timestamp()),
        refresh_token="refresh-1",
        token_url=TOKEN_URL,
    )
    return responses


def test_tracing_credential_is_the_renewed_login(
    expired_login: list[httpx.Response],
) -> None:
    expired_login.append(httpx.Response(200, json={"access_token": "new-access"}))

    assert resolve_tracing_credential() == ("login", "new-access")


def test_tracing_has_no_login_credential_when_the_refresh_token_is_rejected(
    expired_login: list[httpx.Response],
) -> None:
    expired_login.append(httpx.Response(400, json={"error": "invalid_grant"}))

    assert resolve_tracing_credential() is None


def test_trace_batches_are_sent_with_the_renewed_login(
    expired_login: list[httpx.Response],
) -> None:
    expired_login.append(httpx.Response(200, json={"access_token": "new-access"}))

    assert TraceBatchManager().plus_api.api_key == "new-access"


def test_trace_batches_fall_back_to_anonymous_when_offline(
    expired_login: list[httpx.Response], monkeypatch: pytest.MonkeyPatch
) -> None:
    def offline(url: str, *, data: dict[str, str], timeout: float) -> httpx.Response:
        raise httpx.ConnectError("offline")

    monkeypatch.setattr("crewai_core.auth.oauth2.httpx.post", offline)

    assert TraceBatchManager().plus_api.api_key is None
    assert TokenManager().get_refresh_token() == ("refresh-1", TOKEN_URL)
