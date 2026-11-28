"""AMP lookups that read the saved CLI login send the renewed token once it expired."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qsl

from crewai.skills.cache import SkillCacheManager
from crewai.skills.registry import download_skill
from crewai.utilities.agent_utils import load_agent_from_repository
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
    monkeypatch.delenv("CREWAI_ENTERPRISE_ACTION_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(
        "crewai.utilities.agent_utils._create_plus_client_hook", None, raising=False
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

    def handle_request(
        transport: httpx.HTTPTransport, request: httpx.Request
    ) -> httpx.Response:
        assert str(request.url) == TOKEN_URL
        assert dict(parse_qsl(request.read().decode()))["refresh_token"] == "refresh-1"
        return responses.pop(0)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handle_request)
    TokenManager().save_tokens(
        "old-access",
        int((datetime.now() - timedelta(minutes=5)).timestamp()),
        refresh_token="refresh-1",
        token_url=TOKEN_URL,
    )
    return responses


def test_skill_downloads_use_the_renewed_login(
    expired_login: list[httpx.Response], tmp_path: Path
) -> None:
    expired_login.append(httpx.Response(200, json={"access_token": "new-access"}))
    api = MagicMock()
    api.get_skill.side_effect = RuntimeError("stop after the client is built")

    with (
        patch(
            "crewai.skills.registry.SkillCacheManager",
            return_value=SkillCacheManager(cache_root=tmp_path / "cache"),
        ),
        patch("crewai_core.plus_api.PlusAPI", return_value=api) as plus_api,
        pytest.raises(RuntimeError, match="stop after the client is built"),
    ):
        download_skill("acme", "my-skill")

    assert plus_api.call_args.kwargs["api_key"] == "new-access"


def test_repository_agents_load_with_the_renewed_login(
    expired_login: list[httpx.Response],
) -> None:
    expired_login.append(httpx.Response(200, json={"access_token": "new-access"}))
    response = MagicMock(status_code=200)
    response.json.return_value = {"role": "r", "goal": "g", "backstory": "b"}

    with patch(
        "crewai.plus_api.PlusAPI.get_agent", autospec=True, return_value=response
    ) as get_agent:
        attributes = load_agent_from_repository("my-agent")

    assert attributes["role"] == "r"
    client = get_agent.call_args.args[0]
    assert client.api_key == "new-access"


def test_repository_agents_ask_for_login_when_the_refresh_token_is_rejected(
    expired_login: list[httpx.Response],
) -> None:
    from crewai.auth.token import AuthError

    expired_login.append(httpx.Response(400, json={"error": "invalid_grant"}))

    with (
        patch("crewai.plus_api.PlusAPI.get_agent") as get_agent,
        pytest.raises(AuthError, match="make sure you are logged in"),
    ):
        load_agent_from_repository("my-agent")

    get_agent.assert_not_called()
