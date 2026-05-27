"""Tests for AgentCard signature verification during fetch."""

from __future__ import annotations

from typing import Any

from a2a.client.errors import A2AClientHTTPError
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
import httpx
import pytest

from crewai.a2a.utils.agent_card import afetch_agent_card
from crewai.a2a.utils.agent_card_signing import sign_agent_card


def _generate_rsa_keypair() -> tuple[bytes, bytes]:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_pem, public_pem


def _make_agent_card() -> AgentCard:
    return AgentCard(
        name="Test Agent",
        description="Test description",
        url="http://example.com",
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=True, push_notifications=True),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            AgentSkill(
                id="test-skill",
                name="Test Skill",
                description="Test skill description",
                tags=["test"],
            )
        ],
    )


@pytest.mark.asyncio
async def test_afetch_agent_card_verifies_signature_when_key_provided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_pem, public_pem = _generate_rsa_keypair()
    card = _make_agent_card()
    signature = sign_agent_card(card, private_pem, algorithm="RS256")
    signed_card = card.model_copy(update={"signatures": [signature]})

    async def _fake_get(self, url: str, *args: Any, **kwargs: Any) -> httpx.Response:  # noqa: ANN001
        request = httpx.Request("GET", url)
        return httpx.Response(
            200,
            json=signed_card.model_dump(exclude_none=True),
            request=request,
        )

    monkeypatch.setattr(httpx.AsyncClient, "get", _fake_get, raising=True)

    result = await afetch_agent_card(
        endpoint="http://example.com/.well-known/agent-card.json",
        use_cache=False,
        signature_public_key=public_pem,
        signature_algorithms=["RS256"],
    )

    assert result.name == signed_card.name


@pytest.mark.asyncio
async def test_afetch_agent_card_rejects_invalid_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_pem, public_pem = _generate_rsa_keypair()
    other_private_pem, _other_public_pem = _generate_rsa_keypair()
    card = _make_agent_card()

    # Sign with a different key than the one we will verify with.
    signature = sign_agent_card(card, other_private_pem, algorithm="RS256")
    signed_card = card.model_copy(update={"signatures": [signature]})

    async def _fake_get(self, url: str, *args: Any, **kwargs: Any) -> httpx.Response:  # noqa: ANN001
        request = httpx.Request("GET", url)
        return httpx.Response(
            200,
            json=signed_card.model_dump(exclude_none=True),
            request=request,
        )

    monkeypatch.setattr(httpx.AsyncClient, "get", _fake_get, raising=True)

    with pytest.raises(A2AClientHTTPError) as excinfo:
        await afetch_agent_card(
            endpoint="http://example.com/.well-known/agent-card.json",
            use_cache=False,
            signature_public_key=public_pem,
            signature_algorithms=["RS256"],
        )

    assert excinfo.value.status_code == 422
    assert "signature verification failed" in excinfo.value.message.lower()
