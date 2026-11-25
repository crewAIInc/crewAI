"""CLI callers of the saved login, now that an expired login renews itself."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import NoReturn
from unittest.mock import MagicMock, patch

from click.testing import CliRunner
from crewai_cli import create_json_crew
from crewai_cli.cli import logout
from crewai_cli.command import AuthenticationRequiredError, PlusAPIMixin
from crewai_cli.experimental import eval_crew
from crewai_core import lock_store
from crewai_core.auth.oauth2 import Oauth2Settings
from crewai_core.token_manager import TokenManager
import httpx
import pytest


TOKEN_URL = "https://login.example.com/oauth2/token"


@pytest.fixture(autouse=True)
def storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
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
    return tmp_path


@pytest.fixture
def token_endpoint(monkeypatch: pytest.MonkeyPatch) -> list[httpx.Response]:
    """Responses the provider's token endpoint will give, in order."""
    responses: list[httpx.Response] = []

    def post(url: str, *, data: dict[str, str], timeout: float) -> httpx.Response:
        assert url == TOKEN_URL
        assert data["grant_type"] == "refresh_token"
        return responses.pop(0)

    monkeypatch.setattr("crewai_core.auth.oauth2.httpx.post", post)
    return responses


def _save_expired_login() -> None:
    TokenManager().save_tokens(
        "old-access",
        int((datetime.now() - timedelta(minutes=5)).timestamp()),
        refresh_token="refresh-1",
        token_url=TOKEN_URL,
    )


def test_plus_api_commands_use_the_renewed_token(
    token_endpoint: list[httpx.Response],
) -> None:
    _save_expired_login()
    token_endpoint.append(httpx.Response(200, json={"access_token": "new-access"}))

    command = PlusAPIMixin(telemetry=MagicMock())

    assert command.plus_api_client.headers["Authorization"] == "Bearer new-access"


def test_plus_api_commands_ask_to_log_in_when_the_refresh_token_is_rejected(
    token_endpoint: list[httpx.Response], capsys: pytest.CaptureFixture[str]
) -> None:
    _save_expired_login()
    token_endpoint.append(httpx.Response(400, json={"error": "invalid_grant"}))

    with pytest.raises(AuthenticationRequiredError):
        PlusAPIMixin(telemetry=MagicMock())

    assert "Run 'crewai login' to sign up/login." in capsys.readouterr().out


def test_logout_discards_the_refresh_token() -> None:
    _save_expired_login()

    with patch("crewai_cli.cli.Settings"):
        result = CliRunner().invoke(logout, [])

    assert result.exit_code == 0, result.output
    assert TokenManager().get_refresh_token() is None


def test_eval_reads_the_renewed_login(token_endpoint: list[httpx.Response]) -> None:
    _save_expired_login()
    token_endpoint.append(httpx.Response(200, json={"access_token": "new-access"}))

    assert eval_crew.saved_login() == "new-access"


def test_eval_reads_an_unreadable_refresh_reply_as_not_logged_in(
    token_endpoint: list[httpx.Response],
) -> None:
    """Not as an unreadable credential: the saved login itself is fine."""
    _save_expired_login()
    token_endpoint.append(httpx.Response(200, text="<html>Proxy sign-in</html>"))

    assert eval_crew.saved_login() is None


def test_eval_reads_a_lock_that_cannot_be_taken_as_not_logged_in() -> None:
    def unreachable(name: str, *, timeout: float) -> NoReturn:
        raise ConnectionError("Error 61 connecting to 127.0.0.1:1.")

    _save_expired_login()
    lock_store.set_lock_backend(unreachable)
    try:
        assert eval_crew.saved_login() is None
    finally:
        lock_store.set_lock_backend(None)


def test_platform_tools_fetch_the_integration_token_with_the_renewed_login(
    token_endpoint: list[httpx.Response], monkeypatch: pytest.MonkeyPatch
) -> None:
    _save_expired_login()
    token_endpoint.append(httpx.Response(200, json={"access_token": "new-access"}))
    monkeypatch.setattr(
        create_json_crew,
        "Settings",
        lambda: MagicMock(org_uuid="org-1", enterprise_base_url="https://amp.example"),
    )
    client = MagicMock()
    client.get_integration_token.return_value = "integration-token"
    plus_api = MagicMock(return_value=client)
    monkeypatch.setattr(create_json_crew, "PlusAPI", plus_api)

    assert create_json_crew._get_logged_in_platform_token() == "integration-token"
    assert plus_api.call_args.kwargs["api_key"] == "new-access"
