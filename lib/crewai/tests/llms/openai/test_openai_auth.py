"""Tests for OpenAI auth resolution with Codex OAuth compatibility."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
import os
import stat
from urllib.parse import parse_qsl

import httpx
import jwt

from crewai.llms.auth.openai_auth import (
    resolve_codex_oauth_access_token,
    resolve_openai_bearer_token,
    resolve_platform_api_key_from_local_codex,
)


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


def test_resolve_codex_oauth_does_not_require_openai_api_key(monkeypatch, tmp_path):
    access_token = _make_jwt(expires_in_seconds=3600)
    auth_json_path = tmp_path / "auth.json"
    auth_json_path.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": access_token,
                    "refresh_token": "refresh-token",
                },
                "last_refresh": datetime.now(UTC).isoformat(),
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("CREWAI_OPENAI_AUTH_MODE", "oauth_codex")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))

    resolved = resolve_codex_oauth_access_token()
    assert resolved.token == access_token
    assert resolved.source == "codex_auth_json_oauth"
    assert resolved.is_oauth is True


def test_resolve_platform_uses_auth_json_openai_api_key(monkeypatch, tmp_path):
    auth_json_path = tmp_path / "auth.json"
    auth_json_path.write_text(
        json.dumps(
            {
                "OPENAI_API_KEY": "sk-local-platform-key",
                "tokens": {
                    "access_token": _make_jwt(expires_in_seconds=1800),
                    "refresh_token": "refresh-token",
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))

    resolved = resolve_platform_api_key_from_local_codex()
    assert resolved.token == "sk-local-platform-key"
    assert resolved.source == "codex_auth_json_api_key"
    assert resolved.is_oauth is False


def test_resolve_platform_reloads_refresh_token_after_reuse(monkeypatch, tmp_path):
    expired_id = _make_jwt(expires_in_seconds=-300)
    fresh_id = _make_jwt(expires_in_seconds=7200)
    auth_json_path = tmp_path / "auth.json"
    auth_json_path.write_text(
        json.dumps(
            {
                "tokens": {
                    "id_token": expired_id,
                    "refresh_token": "refresh-token-old",
                    "access_token": _make_jwt(expires_in_seconds=1200),
                },
                "last_refresh": (datetime.now(UTC) - timedelta(hours=2)).isoformat(),
            }
        ),
        encoding="utf-8",
    )

    def _handler(request: httpx.Request) -> httpx.Response:
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            body = json.loads(request.content.decode("utf-8"))
            assert body["grant_type"] == "refresh_token"
            if body["refresh_token"] == "refresh-token-old":
                # Simulate another process rotating refresh token.
                latest_payload = json.loads(auth_json_path.read_text(encoding="utf-8"))
                latest_payload["tokens"]["refresh_token"] = "refresh-token-new"
                auth_json_path.write_text(
                    json.dumps(latest_payload), encoding="utf-8"
                )
                return httpx.Response(
                    status_code=401,
                    json={
                        "error": {
                            "message": "Your refresh token has already been used to generate a new access token.",
                            "code": "refresh_token_reused",
                        }
                    },
                )
            assert body["refresh_token"] == "refresh-token-new"
            return httpx.Response(
                status_code=200,
                json={
                    "id_token": fresh_id,
                    "access_token": _make_jwt(expires_in_seconds=3600),
                    "refresh_token": "refresh-token-latest",
                },
            )

        body = dict(parse_qsl(request.content.decode("utf-8")))
        assert body["grant_type"] == "urn:ietf:params:oauth:grant-type:token-exchange"
        assert body["requested_token"] == "openai-api-key"
        assert body["subject_token_type"] == "urn:ietf:params:oauth:token-type:id_token"
        return httpx.Response(
            status_code=200,
            json={"access_token": "sk-from-token-exchange"},
        )

    transport = httpx.MockTransport(_handler)
    with httpx.Client(transport=transport) as client:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("CODEX_HOME", str(tmp_path))
        resolved = resolve_platform_api_key_from_local_codex(http_client=client)

    assert resolved.token == "sk-from-token-exchange"
    assert resolved.source == "codex_token_exchange_api_key"
    updated_payload = json.loads(auth_json_path.read_text(encoding="utf-8"))
    assert updated_payload["tokens"]["refresh_token"] == "refresh-token-latest"
    assert updated_payload["OPENAI_API_KEY"] == "sk-from-token-exchange"
