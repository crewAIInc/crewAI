"""Tests that server-side token/API-key auth uses constant-time comparison.

Regression tests for a timing side-channel: server auth schemes validate
attacker-controlled bearer tokens/API keys from incoming A2A requests. Using a
plain ``!=`` comparison leaks secret length/prefix via response timing.
"""

import hmac

import pytest

from crewai.a2a.auth.server_schemes import APIKeyServerAuth, SimpleTokenAuth


@pytest.mark.asyncio
async def test_simple_token_uses_constant_time_compare(monkeypatch):
    auth = SimpleTokenAuth(token="s3cret-token")

    calls = {"n": 0}
    real = hmac.compare_digest

    def spy(a, b):
        calls["n"] += 1
        return real(a, b)

    monkeypatch.setattr(hmac, "compare_digest", spy)

    user = await auth.authenticate("s3cret-token")
    assert user.token == "s3cret-token"
    assert calls["n"] >= 1, "expected constant-time hmac.compare_digest to be used"


@pytest.mark.asyncio
async def test_api_key_uses_constant_time_compare(monkeypatch):
    auth = APIKeyServerAuth(api_key="my-api-key")

    calls = {"n": 0}
    real = hmac.compare_digest

    def spy(a, b):
        calls["n"] += 1
        return real(a, b)

    monkeypatch.setattr(hmac, "compare_digest", spy)

    user = await auth.authenticate("my-api-key")
    assert user.token == "my-api-key"
    assert calls["n"] >= 1, "expected constant-time hmac.compare_digest to be used"
