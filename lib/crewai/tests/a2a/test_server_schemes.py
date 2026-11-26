"""Tests for A2A server authentication schemes."""

from __future__ import annotations

import hmac

from crewai.a2a.auth.server_schemes import (
    APIKeyServerAuth,
    HTTPException,
    SimpleTokenAuth,
)
from pydantic import SecretStr, ValidationError
import pytest


@pytest.mark.parametrize("empty_token", ["", SecretStr("")])
def test_simple_token_auth_rejects_empty_explicit_token(
    empty_token: str | SecretStr,
) -> None:
    with pytest.raises(ValidationError, match="token must not be empty"):
        SimpleTokenAuth(token=empty_token)


@pytest.mark.asyncio
async def test_simple_token_auth_rejects_empty_env_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTH_TOKEN", "")
    auth = SimpleTokenAuth()

    with pytest.raises(HTTPException) as exc_info:
        await auth.authenticate("")

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Authentication not configured"


@pytest.mark.asyncio
async def test_simple_token_auth_accepts_non_empty_env_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTH_TOKEN", "env-secret")
    auth = SimpleTokenAuth()

    user = await auth.authenticate("env-secret")
    assert user.scheme == "simple_token"
    assert user.token == "env-secret"

    with pytest.raises(HTTPException) as exc_info:
        await auth.authenticate("")
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_simple_token_auth_rejects_unset_env_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AUTH_TOKEN", raising=False)
    auth = SimpleTokenAuth()

    with pytest.raises(HTTPException) as exc_info:
        await auth.authenticate("anything")

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Authentication not configured"


@pytest.mark.asyncio
async def test_simple_token_auth_uses_constant_time_comparison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[bytes, bytes]] = []
    real_compare_digest = hmac.compare_digest

    def spy_compare_digest(a: bytes, b: bytes) -> bool:
        calls.append((a, b))
        return real_compare_digest(a, b)

    monkeypatch.setattr(hmac, "compare_digest", spy_compare_digest)
    auth = SimpleTokenAuth(token="s3cr3t-token")

    user = await auth.authenticate("s3cr3t-token")
    assert user.scheme == "simple_token"
    assert user.token == "s3cr3t-token"

    with pytest.raises(HTTPException) as exc_info:
        await auth.authenticate("wrong-token")
    assert exc_info.value.status_code == 401

    with pytest.raises(HTTPException) as non_ascii_exc:
        await auth.authenticate("s3cr3t-tökén")
    assert non_ascii_exc.value.status_code == 401

    assert len(calls) == 3


@pytest.mark.parametrize("empty_key", ["", SecretStr("")])
def test_api_key_server_auth_rejects_empty_api_key(
    empty_key: str | SecretStr,
) -> None:
    with pytest.raises(ValidationError, match="api_key must not be empty"):
        APIKeyServerAuth(api_key=empty_key)


@pytest.mark.asyncio
async def test_api_key_server_auth_rejects_empty_bypass_via_model_construct() -> None:
    auth = APIKeyServerAuth.model_construct(api_key=SecretStr(""))

    with pytest.raises(HTTPException) as exc_info:
        await auth.authenticate("")

    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_api_key_server_auth_uses_constant_time_comparison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[bytes, bytes]] = []
    real_compare_digest = hmac.compare_digest

    def spy_compare_digest(a: bytes, b: bytes) -> bool:
        calls.append((a, b))
        return real_compare_digest(a, b)

    monkeypatch.setattr(hmac, "compare_digest", spy_compare_digest)
    auth = APIKeyServerAuth(api_key="api-key-123")

    user = await auth.authenticate("api-key-123")
    assert user.scheme == "api_key"
    assert user.token == "api-key-123"

    with pytest.raises(HTTPException) as exc_info:
        await auth.authenticate("bad-key")
    assert exc_info.value.status_code == 401

    with pytest.raises(HTTPException) as non_ascii_exc:
        await auth.authenticate("api-kèy-123")
    assert non_ascii_exc.value.status_code == 401

    assert len(calls) == 3
