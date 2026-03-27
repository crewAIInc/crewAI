"""Tests for TOTP authentication scheme."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pyotp
import pytest

from crewai.a2a.auth.server_schemes import AuthenticatedUser, SimpleTokenAuth
from crewai.a2a.auth.totp_scheme import (
    TOTPClientAuthScheme,
    TOTPServerAuthScheme,
)
from crewai.a2a.auth.client_schemes import BearerTokenAuth


# --- Fixtures ---

SHARED_SEED = pyotp.random_base32()
PEER_SEED_MRPINK = pyotp.random_base32()
PEER_SEED_CHARLIE = pyotp.random_base32()
VALID_TOKEN = "valid-bearer-token"
MRPINK_TOKEN = "mrpink-token"
CHARLIE_TOKEN = "charlie-token"


@pytest.fixture
def shared_seed_scheme() -> TOTPServerAuthScheme:
    """TOTP server scheme with a single shared seed."""
    return TOTPServerAuthScheme(
        delegate=SimpleTokenAuth(token=VALID_TOKEN),
        shared_seed=SHARED_SEED,
    )


@pytest.fixture
def per_peer_scheme() -> TOTPServerAuthScheme:
    """TOTP server scheme with per-peer seeds."""
    return TOTPServerAuthScheme(
        delegate=SimpleTokenAuth(token=MRPINK_TOKEN),
        peer_seeds={
            "mrpink": PEER_SEED_MRPINK,
            "charlie": PEER_SEED_CHARLIE,
        },
        token_to_peer={
            MRPINK_TOKEN: "mrpink",
            CHARLIE_TOKEN: "charlie",
        },
    )


# --- Server Tests ---


@pytest.mark.asyncio
async def test_valid_token_valid_totp(shared_seed_scheme: TOTPServerAuthScheme) -> None:
    """Valid bearer token + valid TOTP → authenticated."""
    totp = pyotp.TOTP(SHARED_SEED)
    code = totp.now()

    result = await shared_seed_scheme.authenticate_with_totp(VALID_TOKEN, code)

    assert isinstance(result, AuthenticatedUser)
    assert result.scheme == "totp"
    assert result.token == VALID_TOKEN


@pytest.mark.asyncio
async def test_valid_token_missing_totp(shared_seed_scheme: TOTPServerAuthScheme) -> None:
    """Valid bearer token + missing TOTP → 401."""
    with pytest.raises(Exception) as exc_info:
        await shared_seed_scheme.authenticate_with_totp(VALID_TOKEN, None)
    assert exc_info.value.status_code == 401
    assert "Missing X-TOTP" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_valid_token_wrong_totp(shared_seed_scheme: TOTPServerAuthScheme) -> None:
    """Valid bearer token + wrong TOTP → 401."""
    with pytest.raises(Exception) as exc_info:
        await shared_seed_scheme.authenticate_with_totp(VALID_TOKEN, "000000")
    assert exc_info.value.status_code == 401
    assert "Invalid TOTP" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_valid_token_expired_totp(shared_seed_scheme: TOTPServerAuthScheme) -> None:
    """Valid bearer token + expired TOTP (outside valid_window) → 401."""
    # Generate a code from 5 minutes ago (well outside valid_window=1 = ±30s)
    import time

    totp = pyotp.TOTP(SHARED_SEED)
    old_time = int(time.time()) - 300  # 5 minutes ago
    code = totp.at(old_time)

    with pytest.raises(Exception) as exc_info:
        await shared_seed_scheme.authenticate_with_totp(VALID_TOKEN, code)
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_invalid_bearer_token(shared_seed_scheme: TOTPServerAuthScheme) -> None:
    """Invalid bearer token → 401 (before TOTP check)."""
    totp = pyotp.TOTP(SHARED_SEED)
    code = totp.now()

    with pytest.raises(Exception) as exc_info:
        await shared_seed_scheme.authenticate_with_totp("bad-token", code)
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_per_peer_seed_lookup(per_peer_scheme: TOTPServerAuthScheme) -> None:
    """Per-peer seed lookup works correctly."""
    totp = pyotp.TOTP(PEER_SEED_MRPINK)
    code = totp.now()

    result = await per_peer_scheme.authenticate_with_totp(MRPINK_TOKEN, code)

    assert isinstance(result, AuthenticatedUser)
    assert result.scheme == "totp"


@pytest.mark.asyncio
async def test_per_peer_wrong_seed_fails(per_peer_scheme: TOTPServerAuthScheme) -> None:
    """Using charlie's TOTP code with mrpink's token → 401."""
    totp = pyotp.TOTP(PEER_SEED_CHARLIE)
    code = totp.now()

    with pytest.raises(Exception) as exc_info:
        await per_peer_scheme.authenticate_with_totp(MRPINK_TOKEN, code)
    assert exc_info.value.status_code == 401


# --- Client Tests ---


@pytest.mark.asyncio
async def test_client_injects_totp_header() -> None:
    """Client scheme correctly injects X-TOTP header."""
    seed = pyotp.random_base32()
    client_scheme = TOTPClientAuthScheme(
        delegate=BearerTokenAuth(token="my-token"),
        seed=seed,
    )

    async with httpx.AsyncClient() as client:
        headers: dict[str, str] = {}
        result = await client_scheme.apply_auth(client, headers)

    assert "X-TOTP" in result
    assert "Authorization" in result
    assert result["Authorization"] == "Bearer my-token"

    # Verify the injected code is valid
    totp = pyotp.TOTP(seed)
    assert totp.verify(result["X-TOTP"], valid_window=1)


@pytest.mark.asyncio
async def test_client_env_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Client scheme reads seed from env var."""
    seed = pyotp.random_base32()
    monkeypatch.setenv("A2A_TOTP_SEED", seed)

    client_scheme = TOTPClientAuthScheme(
        delegate=BearerTokenAuth(token="my-token"),
    )

    async with httpx.AsyncClient() as client:
        headers: dict[str, str] = {}
        result = await client_scheme.apply_auth(client, headers)

    assert "X-TOTP" in result
    totp = pyotp.TOTP(seed)
    assert totp.verify(result["X-TOTP"], valid_window=1)


@pytest.mark.asyncio
async def test_client_no_seed_raises() -> None:
    """Client scheme raises ValueError when no seed configured."""
    client_scheme = TOTPClientAuthScheme(
        delegate=BearerTokenAuth(token="my-token"),
    )

    async with httpx.AsyncClient() as client:
        with pytest.raises(ValueError, match="No TOTP seed configured"):
            await client_scheme.apply_auth(client, {})
