"""Tests that A2A delegation does not blindly trust the remote AgentCard."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from a2a.types import AgentCapabilities, AgentCard, AgentInterface
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
import pytest

from crewai.a2a.utils import delegation
from crewai.a2a.utils.agent_card_signing import sign_agent_card
from crewai.a2a.utils.url_safety import is_same_origin, validate_card_url


ENDPOINT = "https://agent.example.com/a2a"


def _card(url: str = ENDPOINT) -> AgentCard:
    return AgentCard(
        name="remote",
        description="d",
        url=url,
        version="1",
        capabilities=AgentCapabilities(),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[],
    )


def _keys() -> tuple[str, str]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    public = key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    return private, public


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://agent.example.com/other", True),
        ("https://agent.example.com:443/x", True),
        ("https://evil.example.net/a2a", False),
        ("http://agent.example.com/a2a", False),
        ("https://agent.example.com:8443/a2a", False),
        ("grpcs://agent.example.com:50051", True),
        ("evil.example.net:50051", False),
        # Cleartext or non-TLS gRPC is refused when the endpoint is HTTPS.
        ("agent.example.com:50051", False),
        ("grpc://agent.example.com:50051", False),
        ("dns:///agent.example.com:50051", False),
        # Userinfo, dns authorities and unknown schemes never match.
        ("grpcs://agent.example.com@evil.example.net:443", False),
        ("dns://10.0.0.1/agent.example.com:443", False),
        ("unix:///var/run/agent.sock", False),
    ],
)
def test_is_same_origin(url: str, expected: bool) -> None:
    assert is_same_origin(url, ENDPOINT) is expected


@pytest.mark.parametrize(
    "url,expected",
    [
        ("localhost:50051", True),
        ("grpc://localhost:50051", True),
        ("dns:///localhost:50051", True),
        ("dns:///other.example.net:50051", False),
        ("grpc://other.example.net:50051", False),
    ],
)
def test_grpc_targets_on_cleartext_endpoint(url: str, expected: bool) -> None:
    assert is_same_origin(url, "http://localhost:8000") is expected


@pytest.mark.asyncio
async def test_cross_origin_card_url_is_rejected() -> None:
    with pytest.raises(ValueError, match="does not match"):
        await validate_card_url("https://169.254.169.254/latest", ENDPOINT)


@pytest.mark.asyncio
async def test_public_endpoint_resolving_internal_is_rejected() -> None:
    with patch(
        "crewai.a2a.utils.url_safety.is_internal_host",
        new=AsyncMock(side_effect=lambda h: h == "agent.example.com"),
    ), patch(
        "crewai.a2a.utils.url_safety.is_same_origin", return_value=True
    ):
        with pytest.raises(ValueError, match="internal"):
            await validate_card_url(ENDPOINT, "https://other.example.com")


@pytest.mark.asyncio
async def test_local_endpoint_is_allowed() -> None:
    await validate_card_url("http://127.0.0.1:9000/a", "http://127.0.0.1:9000")


def test_valid_signature_passes_and_tampering_fails() -> None:
    private, public = _keys()
    card = _card()
    card.signatures = [sign_agent_card(card, private_key=private)]
    delegation._verify_card_signature(card, public, None)

    card.name = "tampered"
    with pytest.raises(ValueError, match="signature"):
        delegation._verify_card_signature(card, public, None)


def test_unsigned_card_fails_when_key_configured() -> None:
    _, public = _keys()
    with pytest.raises(ValueError, match="signature"):
        delegation._verify_card_signature(_card(), public, None)


@pytest.mark.asyncio
async def test_delegation_never_builds_client_for_foreign_card_url() -> None:
    card = _card("https://evil.example.net/a2a")
    with patch.object(
        delegation, "_afetch_agent_card_cached", AsyncMock(return_value=card)
    ), patch.object(delegation, "_create_a2a_client") as create_client:
        with pytest.raises(ValueError, match="does not match"):
            await delegation._aexecute_a2a_delegation_impl(
                endpoint=ENDPOINT,
                auth=None,
                timeout=5,
                task_description="t",
                context=None,
                context_id=None,
                task_id=None,
                reference_task_ids=None,
                metadata=None,
                extensions=None,
                conversation_history=[],
                is_multiturn=False,
                turn_number=1,
                agent_branch=None,
                agent_id=None,
                agent_role=None,
                response_model=None,
                updates=None,
            )
    create_client.assert_not_called()


def _delegate(card: AgentCard) -> object:
    return delegation._aexecute_a2a_delegation_impl(
        endpoint=ENDPOINT,
        auth=None,
        timeout=5,
        task_description="t",
        context=None,
        context_id=None,
        task_id=None,
        reference_task_ids=None,
        metadata=None,
        extensions=None,
        conversation_history=[],
        is_multiturn=False,
        turn_number=1,
        agent_branch=None,
        agent_id=None,
        agent_role=None,
        response_model=None,
        updates=None,
    )


@pytest.mark.asyncio
async def test_off_origin_additional_interface_never_reaches_client() -> None:
    """A card repeating a transport off-origin must not override the safe URL."""
    card = _card()
    card.additional_interfaces = [
        AgentInterface(transport="JSONRPC", url="https://evil.example.net/a2a"),
        AgentInterface(transport="GRPC", url="grpcs://agent.example.com:50051"),
    ]
    seen: dict[str, AgentCard] = {}

    def fake_client(**kwargs: object) -> None:
        seen["card"] = kwargs["agent_card"]  # type: ignore[assignment]
        raise RuntimeError("stop")

    with patch.object(
        delegation, "_afetch_agent_card_cached", AsyncMock(return_value=card)
    ), patch.object(delegation, "_create_a2a_client", side_effect=fake_client):
        with pytest.raises(RuntimeError, match="stop"):
            await _delegate(card)  # type: ignore[misc]

    urls = [i.url for i in seen["card"].additional_interfaces or []]
    assert urls == ["grpcs://agent.example.com:50051"]
