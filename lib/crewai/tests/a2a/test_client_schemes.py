"""Tests for client authentication schemes."""

from __future__ import annotations

import asyncio

from crewai.a2a.auth.client_schemes import APIKeyAuth


def _apply(auth: APIKeyAuth, headers: dict[str, str]) -> dict[str, str]:
    return asyncio.run(auth.apply_auth(None, headers))  # type: ignore[arg-type]


def test_cookie_location_preserves_existing_cookies() -> None:
    auth = APIKeyAuth(api_key="secret", location="cookie", name="api_key")
    headers = {"Cookie": "session=abc123; theme=dark"}

    result = _apply(auth, headers)

    assert result["Cookie"] == "session=abc123; theme=dark; api_key=secret"


def test_cookie_location_sets_cookie_when_absent() -> None:
    auth = APIKeyAuth(api_key="secret", location="cookie", name="api_key")

    result = _apply(auth, {})

    assert result == {"Cookie": "api_key=secret"}


def test_cookie_location_reuses_lowercase_header_name() -> None:
    auth = APIKeyAuth(api_key="secret", location="cookie", name="api_key")
    headers = {"cookie": "session=abc123"}

    result = _apply(auth, headers)

    assert result == {"cookie": "session=abc123; api_key=secret"}
