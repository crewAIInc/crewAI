"""Tests for OpenAI auth resolution with Codex OAuth compatibility."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
import os
import stat

import httpx
import jwt

from crewai.llms.auth.openai_auth import resolve_openai_bearer_token


def _make_jwt(expires_in_seconds: int) -> str:
    """Create a signed JWT for expiry tests."""
    exp = int((datetime.now(UTC) + timedelta(seconds=expires_in_seconds)).timestamp())
    return jwt.encode({"exp": exp, "sub": "test-user"}, "test-secret", algorithm="HS256")


def test_resolve_prefers_explicit_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-openai-key")
    monkeypatch.setenv("OPENAI_OAUTH_ACCESS_TOKEN", "env-oauth-token")
    resolved = resolve_openai_bearer_token(explicit_api_key="explicit-openai-key")
    assert resolved.token == "explicit-openai-key"
    assert resolved.source == "explicit_api_key"
    assert resolved.is_oauth is False


def test_resolve_uses_env_openai_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-openai-key")
    monkeypatch.delenv("OPENAI_OAUTH_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("CREWAI_OPENAI_AUTH_MODE", raising=False)
    resolved = resolve_openai_bearer_token()
    assert resolved.token == "env-openai-key"
    assert resolved.source == "env_openai_api_key"
    assert resolved.is_oauth is False


def test_resolve_uses_env_oauth_access_token(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_OAUTH_ACCESS_TOKEN", "oauth-access-token")
    monkeypatch.setenv("CREWAI_OPENAI_AUTH_MODE", "oauth_token")
    resolved = resolve_openai_bearer_token()
    assert resolved.token == "oauth-access-token"
    assert resolved.source == "env_oauth_access_token"
    assert resolved.is_oauth is True


def test_resolve_uses_codex_auth_json_access_token(monkeypatch, tmp_path):
    access_token = _make_jwt(expires_in_seconds=3600)
    auth_json_path = tmp_path / "auth.json"
    auth_json_path.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": access_token,
                    "refresh_token": "refresh-token",
                    "account_id": "acct_test",
                },
                "last_refresh": datetime.now(UTC).isoformat(),
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_OAUTH_ACCESS_TOKEN", raising=False)
    monkeypatch.setenv("CREWAI_OPENAI_AUTH_MODE", "oauth_codex")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))

    resolved = resolve_openai_bearer_token()

    assert resolved.token == access_token
    assert resolved.source == "codex_auth_json_oauth"
    assert resolved.account_id == "acct_test"
    assert resolved.is_oauth is True


def test_refresh_flow_updates_auth_json(monkeypatch, tmp_path):
    expired_token = _make_jwt(expires_in_seconds=-120)
    refreshed_access_token = _make_jwt(expires_in_seconds=7200)
    auth_json_path = tmp_path / "auth.json"
    auth_json_path.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": expired_token,
                    "refresh_token": "refresh-token-old",
                    "account_id": "acct_old",
                },
                "last_refresh": (datetime.now(UTC) - timedelta(hours=2)).isoformat(),
            }
        ),
        encoding="utf-8",
    )

    def _handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == "https://auth.openai.com/oauth/token"
        body = json.loads(request.content.decode("utf-8"))
        assert body["grant_type"] == "refresh_token"
        assert body["refresh_token"] == "refresh-token-old"
        return httpx.Response(
            status_code=200,
            json={
                "access_token": refreshed_access_token,
                "refresh_token": "refresh-token-new",
                "id_token": "id-token-new",
            },
        )

    transport = httpx.MockTransport(_handler)
    with httpx.Client(transport=transport) as client:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_OAUTH_ACCESS_TOKEN", raising=False)
        monkeypatch.setenv("CREWAI_OPENAI_AUTH_MODE", "oauth_codex")
        monkeypatch.setenv("CODEX_HOME", str(tmp_path))

        resolved = resolve_openai_bearer_token(http_client=client)

    assert resolved.token == refreshed_access_token
    assert resolved.source == "codex_auth_json_oauth"

    updated_payload = json.loads(auth_json_path.read_text(encoding="utf-8"))
    updated_tokens = updated_payload["tokens"]
    assert updated_tokens["access_token"] == refreshed_access_token
    assert updated_tokens["refresh_token"] == "refresh-token-new"
    assert updated_tokens["id_token"] == "id-token-new"
    assert updated_tokens["account_id"] == "acct_old"
    assert updated_payload["last_refresh"]

    if os.name != "nt":
        file_mode = stat.S_IMODE(auth_json_path.stat().st_mode)
        assert file_mode == 0o600
