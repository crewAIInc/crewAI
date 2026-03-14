"""Tests for A2A agent card JWS signing utilities."""

from __future__ import annotations

import base64
import json

import pytest
from a2a.types import AgentCapabilities, AgentCard, AgentCardSignature, AgentSkill
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from pydantic import SecretStr

from crewai.a2a.utils.agent_card_signing import (
    _base64url_encode,
    _normalize_private_key,
    _serialize_agent_card,
    get_key_id_from_signature,
    sign_agent_card,
    verify_agent_card_signature,
)


# ---------------------------------------------------------------------------
# Fixtures: dynamically generated keys (no hardcoded secrets)
# ---------------------------------------------------------------------------


@pytest.fixture()
def rsa_private_key_pem() -> bytes:
    """Generate a fresh RSA private key in PEM format."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )


@pytest.fixture()
def rsa_public_key_pem(rsa_private_key_pem: bytes) -> bytes:
    """Derive the RSA public key from the private key."""
    from cryptography.hazmat.primitives.serialization import load_pem_private_key

    key = load_pem_private_key(rsa_private_key_pem, password=None)
    return key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )


@pytest.fixture()
def ec_private_key_pem() -> bytes:
    """Generate a fresh EC (P-256) private key in PEM format."""
    key = ec.generate_private_key(ec.SECP256R1())
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )


@pytest.fixture()
def ec_public_key_pem(ec_private_key_pem: bytes) -> bytes:
    """Derive the EC public key from the private key."""
    from cryptography.hazmat.primitives.serialization import load_pem_private_key

    key = load_pem_private_key(ec_private_key_pem, password=None)
    return key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )


@pytest.fixture()
def sample_agent_card() -> AgentCard:
    """Create a minimal AgentCard for testing."""
    return AgentCard(
        name="Test Agent",
        description="A test agent for signing",
        url="http://localhost:8000",
        version="1.0.0",
        skills=[
            AgentSkill(
                id="test-skill",
                name="Test Skill",
                description="A test skill",
                tags=["test"],
            )
        ],
        capabilities=AgentCapabilities(streaming=False, pushNotifications=False),
        defaultInputModes=["text/plain"],
        defaultOutputModes=["text/plain"],
    )


# ---------------------------------------------------------------------------
# Tests: _normalize_private_key
# ---------------------------------------------------------------------------


class TestNormalizePrivateKey:
    """Tests for the _normalize_private_key helper."""

    def test_bytes_input_returns_same_bytes(self, rsa_private_key_pem: bytes) -> None:
        result = _normalize_private_key(rsa_private_key_pem)
        assert result == rsa_private_key_pem

    def test_str_input_returns_encoded_bytes(self, rsa_private_key_pem: bytes) -> None:
        key_str = rsa_private_key_pem.decode()
        result = _normalize_private_key(key_str)
        assert result == rsa_private_key_pem

    def test_secret_str_input_returns_bytes(self, rsa_private_key_pem: bytes) -> None:
        secret = SecretStr(rsa_private_key_pem.decode())
        result = _normalize_private_key(secret)
        assert result == rsa_private_key_pem


# ---------------------------------------------------------------------------
# Tests: _serialize_agent_card
# ---------------------------------------------------------------------------


class TestSerializeAgentCard:
    """Tests for the _serialize_agent_card helper."""

    def test_returns_valid_json(self, sample_agent_card: AgentCard) -> None:
        result = _serialize_agent_card(sample_agent_card)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_excludes_signatures_field(self, sample_agent_card: AgentCard) -> None:
        result = _serialize_agent_card(sample_agent_card)
        parsed = json.loads(result)
        assert "signatures" not in parsed

    def test_deterministic_output(self, sample_agent_card: AgentCard) -> None:
        """Serialization should produce the same output on repeated calls."""
        result1 = _serialize_agent_card(sample_agent_card)
        result2 = _serialize_agent_card(sample_agent_card)
        assert result1 == result2

    def test_sorted_keys(self, sample_agent_card: AgentCard) -> None:
        result = _serialize_agent_card(sample_agent_card)
        parsed = json.loads(result)
        keys = list(parsed.keys())
        assert keys == sorted(keys)

    def test_compact_separators(self, sample_agent_card: AgentCard) -> None:
        """Output should use compact separators (no spaces after : or ,)."""
        result = _serialize_agent_card(sample_agent_card)
        # Compact JSON should not have ": " or ", " patterns at the top level
        assert ": " not in result or result.count(": ") == 0
        # More specifically, there should be no space after colon
        # (json.dumps with separators=(",",":") ensures this)


# ---------------------------------------------------------------------------
# Tests: _base64url_encode
# ---------------------------------------------------------------------------


class TestBase64urlEncode:
    """Tests for the _base64url_encode helper."""

    def test_bytes_input(self) -> None:
        result = _base64url_encode(b"hello")
        assert isinstance(result, str)
        # Verify no padding
        assert "=" not in result

    def test_str_input(self) -> None:
        result = _base64url_encode("hello")
        assert isinstance(result, str)

    def test_no_padding(self) -> None:
        """Output should have no base64 padding characters."""
        # Use input that normally produces padding
        result = _base64url_encode(b"a")
        assert "=" not in result

    def test_url_safe(self) -> None:
        """Output should not contain + or / characters."""
        # Use input that may produce + or / in standard base64
        data = bytes(range(256))
        result = _base64url_encode(data)
        assert "+" not in result
        assert "/" not in result

    def test_roundtrip(self) -> None:
        """Encoded data should be decodable back to original."""
        original = b"test payload data"
        encoded = _base64url_encode(original)
        # Add padding back for decoding
        padding = 4 - len(encoded) % 4
        if padding != 4:
            encoded += "=" * padding
        decoded = base64.urlsafe_b64decode(encoded)
        assert decoded == original


# ---------------------------------------------------------------------------
# Tests: sign_agent_card (RSA)
# ---------------------------------------------------------------------------


class TestSignAgentCardRSA:
    """Tests for signing agent cards with RSA keys."""

    def test_returns_agent_card_signature(
        self, sample_agent_card: AgentCard, rsa_private_key_pem: bytes
    ) -> None:
        result = sign_agent_card(sample_agent_card, rsa_private_key_pem)
        assert isinstance(result, AgentCardSignature)

    def test_signature_has_protected_header(
        self, sample_agent_card: AgentCard, rsa_private_key_pem: bytes
    ) -> None:
        result = sign_agent_card(sample_agent_card, rsa_private_key_pem)
        assert result.protected is not None
        assert len(result.protected) > 0

    def test_signature_has_signature_value(
        self, sample_agent_card: AgentCard, rsa_private_key_pem: bytes
    ) -> None:
        result = sign_agent_card(sample_agent_card, rsa_private_key_pem)
        assert result.signature is not None
        assert len(result.signature) > 0

    def test_with_key_id(
        self, sample_agent_card: AgentCard, rsa_private_key_pem: bytes
    ) -> None:
        result = sign_agent_card(
            sample_agent_card, rsa_private_key_pem, key_id="test-key-1"
        )
        assert result.header is not None
        assert result.header["kid"] == "test-key-1"

    def test_without_key_id(
        self, sample_agent_card: AgentCard, rsa_private_key_pem: bytes
    ) -> None:
        result = sign_agent_card(sample_agent_card, rsa_private_key_pem)
        assert result.header is None

    def test_accepts_str_key(
        self, sample_agent_card: AgentCard, rsa_private_key_pem: bytes
    ) -> None:
        result = sign_agent_card(
            sample_agent_card, rsa_private_key_pem.decode()
        )
        assert isinstance(result, AgentCardSignature)

    def test_accepts_secret_str_key(
        self, sample_agent_card: AgentCard, rsa_private_key_pem: bytes
    ) -> None:
        secret = SecretStr(rsa_private_key_pem.decode())
        result = sign_agent_card(sample_agent_card, secret)
        assert isinstance(result, AgentCardSignature)

    def test_protected_header_contains_typ(
        self, sample_agent_card: AgentCard, rsa_private_key_pem: bytes
    ) -> None:
        result = sign_agent_card(sample_agent_card, rsa_private_key_pem)
        # Decode protected header
        protected = result.protected
        padding = 4 - len(protected) % 4
        if padding != 4:
            protected += "=" * padding
        header = json.loads(base64.urlsafe_b64decode(protected))
        assert header.get("typ") == "JWS"

    def test_protected_header_contains_algorithm(
        self, sample_agent_card: AgentCard, rsa_private_key_pem: bytes
    ) -> None:
        result = sign_agent_card(
            sample_agent_card, rsa_private_key_pem, algorithm="RS256"
        )
        protected = result.protected
        padding = 4 - len(protected) % 4
        if padding != 4:
            protected += "=" * padding
        header = json.loads(base64.urlsafe_b64decode(protected))
        assert header.get("alg") == "RS256"


# ---------------------------------------------------------------------------
# Tests: sign_agent_card (EC)
# ---------------------------------------------------------------------------


class TestSignAgentCardEC:
    """Tests for signing agent cards with EC keys."""

    def test_sign_with_es256(
        self, sample_agent_card: AgentCard, ec_private_key_pem: bytes
    ) -> None:
        result = sign_agent_card(
            sample_agent_card, ec_private_key_pem, algorithm="ES256"
        )
        assert isinstance(result, AgentCardSignature)

    def test_protected_header_has_es256(
        self, sample_agent_card: AgentCard, ec_private_key_pem: bytes
    ) -> None:
        result = sign_agent_card(
            sample_agent_card, ec_private_key_pem, algorithm="ES256"
        )
        protected = result.protected
        padding = 4 - len(protected) % 4
        if padding != 4:
            protected += "=" * padding
        header = json.loads(base64.urlsafe_b64decode(protected))
        assert header.get("alg") == "ES256"


# ---------------------------------------------------------------------------
# Tests: verify_agent_card_signature
# ---------------------------------------------------------------------------


class TestVerifyAgentCardSignature:
    """Tests for verifying agent card signatures."""

    def test_valid_rsa_signature(
        self,
        sample_agent_card: AgentCard,
        rsa_private_key_pem: bytes,
        rsa_public_key_pem: bytes,
    ) -> None:
        sig = sign_agent_card(sample_agent_card, rsa_private_key_pem)
        assert verify_agent_card_signature(
            sample_agent_card, sig, rsa_public_key_pem
        )

    def test_valid_ec_signature(
        self,
        sample_agent_card: AgentCard,
        ec_private_key_pem: bytes,
        ec_public_key_pem: bytes,
    ) -> None:
        sig = sign_agent_card(
            sample_agent_card, ec_private_key_pem, algorithm="ES256"
        )
        assert verify_agent_card_signature(
            sample_agent_card, sig, ec_public_key_pem
        )

    def test_invalid_signature_returns_false(
        self,
        sample_agent_card: AgentCard,
        rsa_private_key_pem: bytes,
    ) -> None:
        sig = sign_agent_card(sample_agent_card, rsa_private_key_pem)
        # Generate a different key pair for verification (wrong key)
        wrong_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        wrong_pub = wrong_key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        assert not verify_agent_card_signature(
            sample_agent_card, sig, wrong_pub
        )

    def test_tampered_card_fails_verification(
        self,
        sample_agent_card: AgentCard,
        rsa_private_key_pem: bytes,
        rsa_public_key_pem: bytes,
    ) -> None:
        sig = sign_agent_card(sample_agent_card, rsa_private_key_pem)
        # Tamper with the card
        tampered_card = sample_agent_card.model_copy(
            update={"description": "Tampered description"}
        )
        assert not verify_agent_card_signature(
            tampered_card, sig, rsa_public_key_pem
        )

    def test_corrupted_signature_returns_false(
        self,
        sample_agent_card: AgentCard,
        rsa_private_key_pem: bytes,
        rsa_public_key_pem: bytes,
    ) -> None:
        sig = sign_agent_card(sample_agent_card, rsa_private_key_pem)
        corrupted = AgentCardSignature(
            protected=sig.protected,
            signature="corrupted_signature_value",
            header=sig.header,
        )
        assert not verify_agent_card_signature(
            sample_agent_card, corrupted, rsa_public_key_pem
        )

    def test_accepts_str_public_key(
        self,
        sample_agent_card: AgentCard,
        rsa_private_key_pem: bytes,
        rsa_public_key_pem: bytes,
    ) -> None:
        sig = sign_agent_card(sample_agent_card, rsa_private_key_pem)
        assert verify_agent_card_signature(
            sample_agent_card, sig, rsa_public_key_pem.decode()
        )

    def test_custom_algorithms_list(
        self,
        sample_agent_card: AgentCard,
        rsa_private_key_pem: bytes,
        rsa_public_key_pem: bytes,
    ) -> None:
        sig = sign_agent_card(
            sample_agent_card, rsa_private_key_pem, algorithm="RS256"
        )
        assert verify_agent_card_signature(
            sample_agent_card, sig, rsa_public_key_pem, algorithms=["RS256"]
        )

    def test_algorithm_mismatch_returns_false(
        self,
        sample_agent_card: AgentCard,
        rsa_private_key_pem: bytes,
        rsa_public_key_pem: bytes,
    ) -> None:
        sig = sign_agent_card(
            sample_agent_card, rsa_private_key_pem, algorithm="RS256"
        )
        # Only allow ES256 for verification - should fail
        assert not verify_agent_card_signature(
            sample_agent_card, sig, rsa_public_key_pem, algorithms=["ES256"]
        )

    def test_sign_and_verify_with_key_id(
        self,
        sample_agent_card: AgentCard,
        rsa_private_key_pem: bytes,
        rsa_public_key_pem: bytes,
    ) -> None:
        sig = sign_agent_card(
            sample_agent_card,
            rsa_private_key_pem,
            key_id="my-key-id",
        )
        assert verify_agent_card_signature(
            sample_agent_card, sig, rsa_public_key_pem
        )


# ---------------------------------------------------------------------------
# Tests: get_key_id_from_signature
# ---------------------------------------------------------------------------


class TestGetKeyIdFromSignature:
    """Tests for extracting key IDs from signatures."""

    def test_key_id_from_unprotected_header(self) -> None:
        sig = AgentCardSignature(
            protected="eyJ0eXAiOiJKV1MiLCJhbGciOiJSUzI1NiJ9",
            signature="dummy",
            header={"kid": "unprotected-key"},
        )
        assert get_key_id_from_signature(sig) == "unprotected-key"

    def test_key_id_from_protected_header(
        self, sample_agent_card: AgentCard, rsa_private_key_pem: bytes
    ) -> None:
        sig = sign_agent_card(
            sample_agent_card, rsa_private_key_pem, key_id="protected-key"
        )
        # Remove unprotected header to test protected header extraction
        sig_no_header = AgentCardSignature(
            protected=sig.protected,
            signature=sig.signature,
            header=None,
        )
        assert get_key_id_from_signature(sig_no_header) == "protected-key"

    def test_no_key_id_returns_none(
        self, sample_agent_card: AgentCard, rsa_private_key_pem: bytes
    ) -> None:
        sig = sign_agent_card(sample_agent_card, rsa_private_key_pem)
        assert get_key_id_from_signature(sig) is None

    def test_unprotected_header_takes_precedence(
        self, sample_agent_card: AgentCard, rsa_private_key_pem: bytes
    ) -> None:
        """When both headers have kid, unprotected header wins."""
        sig = sign_agent_card(
            sample_agent_card, rsa_private_key_pem, key_id="protected-id"
        )
        # Override unprotected header with different kid
        sig_with_override = AgentCardSignature(
            protected=sig.protected,
            signature=sig.signature,
            header={"kid": "unprotected-id"},
        )
        assert get_key_id_from_signature(sig_with_override) == "unprotected-id"

    def test_invalid_protected_header_returns_none(self) -> None:
        sig = AgentCardSignature(
            protected="not-valid-base64!!!",
            signature="dummy",
            header=None,
        )
        assert get_key_id_from_signature(sig) is None

    def test_empty_header_dict(self) -> None:
        sig = AgentCardSignature(
            protected="eyJ0eXAiOiJKV1MiLCJhbGciOiJSUzI1NiJ9",
            signature="dummy",
            header={},
        )
        # No kid in empty header, should fall through to protected header
        result = get_key_id_from_signature(sig)
        # Protected header {"typ":"JWS","alg":"RS256"} has no kid
        assert result is None


# ---------------------------------------------------------------------------
# Tests: no hardcoded credentials in source
# ---------------------------------------------------------------------------


class TestNoHardcodedCredentials:
    """Ensure the signing module does not contain hardcoded private keys."""

    def test_no_begin_private_key_in_source(self) -> None:
        """The source file must not contain actual PEM key headers."""
        import inspect

        import crewai.a2a.utils.agent_card_signing as module

        source = inspect.getsource(module)
        assert "-----BEGIN PRIVATE KEY-----" not in source
        assert "-----BEGIN RSA PRIVATE KEY-----" not in source
        assert "-----BEGIN EC PRIVATE KEY-----" not in source
