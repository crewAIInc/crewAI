"""AgentCard JWS signing utilities.

This module provides functions for signing and verifying AgentCards using
JSON Web Signatures (JWS) as per RFC 7515. Signed agent cards allow clients
to verify the authenticity and integrity of agent card information.

Example:
    >>> from crewai.a2a.utils.agent_card_signing import sign_agent_card
    >>> signature = sign_agent_card(agent_card, private_key_pem, key_id="key-1")
    >>> card_with_sig = card.model_copy(update={"signatures": [signature]})
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any, Literal

from a2a.types import AgentCard, AgentCardSignature
import jwt
from pydantic import SecretStr


logger = logging.getLogger(__name__)


SigningAlgorithm = Literal[
    "RS256", "RS384", "RS512", "ES256", "ES384", "ES512", "PS256", "PS384", "PS512"
]


def _normalize_private_key(private_key: str | bytes | SecretStr) -> bytes:
    """Normalize private key to bytes format.

    Args:
        private_key: PEM-encoded private key as string, bytes, or SecretStr.

    Returns:
        Private key as bytes.
    """
    if isinstance(private_key, SecretStr):
        private_key = private_key.get_secret_value()
    if isinstance(private_key, str):
        private_key = private_key.encode()
    return private_key


def _serialize_agent_card(agent_card: AgentCard) -> str:
    """Serialize AgentCard to canonical JSON for signing.

    Excludes the signatures field to avoid circular reference during signing.
    Uses sorted keys and compact separators for deterministic output.

    Args:
        agent_card: The AgentCard to serialize.

    Returns:
        Canonical JSON string representation.
    """
    card_dict = agent_card.model_dump(exclude={"signatures"}, exclude_none=True)
    return json.dumps(card_dict, sort_keys=True, separators=(",", ":"))


def _base64url_encode(data: bytes | str) -> str:
    """Encode data to URL-safe base64 without padding.

    Args:
        data: Data to encode.

    Returns:
        URL-safe base64 encoded string without padding.
    """
    if isinstance(data, str):
        data = data.encode()
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def sign_agent_card(
    agent_card: AgentCard,
    private_key: str | bytes | SecretStr,
    key_id: str | None = None,
    algorithm: SigningAlgorithm = "RS256",
) -> AgentCardSignature:
    """Sign an AgentCard using JWS (RFC 7515).

    Creates a detached JWS signature for the AgentCard. The signature covers
    all fields except the signatures field itself.

    Args:
        agent_card: The AgentCard to sign.
        private_key: PEM-encoded private key (RSA, EC, or RSA-PSS).
        key_id: Optional key identifier for the JWS header (kid claim).
        algorithm: Signing algorithm (RS256, ES256, PS256, etc.).

    Returns:
        AgentCardSignature with protected header and signature.

    Raises:
        jwt.exceptions.InvalidKeyError: If the private key is invalid.
        ValueError: If the algorithm is not supported for the key type.

    Example:
        >>> signature = sign_agent_card(
        ...     agent_card,
        ...     private_key_pem="-----BEGIN PRIVATE KEY-----...",
        ...     key_id="my-key-id",
        ... )
    """
    key_bytes = _normalize_private_key(private_key)
    payload = _serialize_agent_card(agent_card)

    protected_header: dict[str, Any] = {"typ": "JWS"}
    if key_id:
        protected_header["kid"] = key_id

    jws_token = jwt.api_jws.encode(
        payload.encode(),
        key_bytes,
        algorithm=algorithm,
        headers=protected_header,
    )

    parts = jws_token.split(".")
    protected_b64 = parts[0]
    signature_b64 = parts[2]

    header: dict[str, Any] | None = None
    if key_id:
        header = {"kid": key_id}

    return AgentCardSignature(
        protected=protected_b64,
        signature=signature_b64,
        header=header,
    )


def verify_agent_card_signature(
    agent_card: AgentCard,
    signature: AgentCardSignature,
    public_key: str | bytes,
    algorithms: list[str] | None = None,
) -> bool:
    """Verify an AgentCard JWS signature.

    Validates that the signature was created with the corresponding private key
    and that the AgentCard content has not been modified.

    Args:
        agent_card: The AgentCard to verify.
        signature: The AgentCardSignature to validate.
        public_key: PEM-encoded public key (RSA, EC, or RSA-PSS).
        algorithms: List of allowed algorithms. Defaults to common asymmetric algorithms.

    Returns:
        True if signature is valid, False otherwise.

    Example:
        >>> is_valid = verify_agent_card_signature(
        ...     agent_card, signature, public_key_pem="-----BEGIN PUBLIC KEY-----..."
        ... )
    """
    if algorithms is None:
        algorithms = [
            "RS256",
            "RS384",
            "RS512",
            "ES256",
            "ES384",
            "ES512",
            "PS256",
            "PS384",
            "PS512",
        ]

    if isinstance(public_key, str):
        public_key = public_key.encode()

    payload = _serialize_agent_card(agent_card)
    payload_b64 = _base64url_encode(payload)
    jws_token = f"{signature.protected}.{payload_b64}.{signature.signature}"

    try:
        jwt.api_jws.decode(
            jws_token,
            public_key,
            algorithms=algorithms,
        )
        return True
    except jwt.InvalidSignatureError:
        logger.debug(
            "AgentCard signature verification failed",
            extra={"reason": "invalid_signature"},
        )
        return False
    except jwt.DecodeError as e:
        logger.debug(
            "AgentCard signature verification failed",
            extra={"reason": "decode_error", "error": str(e)},
        )
        return False
    except jwt.InvalidAlgorithmError as e:
        logger.debug(
            "AgentCard signature verification failed",
            extra={"reason": "algorithm_error", "error": str(e)},
        )
        return False


def get_key_id_from_signature(signature: AgentCardSignature) -> str | None:
    """Extract the key ID (kid) from an AgentCardSignature.

    Checks both the unprotected header and the protected header for the kid claim.

    Args:
        signature: The AgentCardSignature to extract from.

    Returns:
        The key ID if present, None otherwise.
    """
    if signature.header and "kid" in signature.header:
        kid: str = signature.header["kid"]
        return kid

    try:
        protected = signature.protected
        padding_needed = 4 - (len(protected) % 4)
        if padding_needed != 4:
            protected += "=" * padding_needed

        protected_json = base64.urlsafe_b64decode(protected).decode()
        protected_header: dict[str, Any] = json.loads(protected_json)
        return protected_header.get("kid")
    except (ValueError, json.JSONDecodeError):
        return None
