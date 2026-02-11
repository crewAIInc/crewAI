"""Webhook signature configuration for push notifications."""

from __future__ import annotations

from enum import Enum
import secrets

from pydantic import BaseModel, Field, SecretStr


class WebhookSignatureMode(str, Enum):
    """Signature mode for webhook push notifications."""

    NONE = "none"
    HMAC_SHA256 = "hmac_sha256"


class WebhookSignatureConfig(BaseModel):
    """Configuration for webhook signature verification.

    Provides cryptographic integrity verification and replay attack protection
    for A2A push notifications.

    Attributes:
        mode: Signature mode (none or hmac_sha256).
        secret: Shared secret for HMAC computation (required for hmac_sha256 mode).
        timestamp_tolerance_seconds: Max allowed age of timestamps for replay protection.
        header_name: HTTP header name for the signature.
        timestamp_header_name: HTTP header name for the timestamp.
    """

    mode: WebhookSignatureMode = Field(
        default=WebhookSignatureMode.NONE,
        description="Signature verification mode",
    )
    secret: SecretStr | None = Field(
        default=None,
        description="Shared secret for HMAC computation",
    )
    timestamp_tolerance_seconds: int = Field(
        default=300,
        ge=0,
        description="Max allowed timestamp age in seconds (5 min default)",
    )
    header_name: str = Field(
        default="X-A2A-Signature",
        description="HTTP header name for the signature",
    )
    timestamp_header_name: str = Field(
        default="X-A2A-Signature-Timestamp",
        description="HTTP header name for the timestamp",
    )

    @classmethod
    def generate_secret(cls, length: int = 32) -> str:
        """Generate a cryptographically secure random secret.

        Args:
            length: Number of random bytes to generate (default 32).

        Returns:
            URL-safe base64-encoded secret string.
        """
        return secrets.token_urlsafe(length)

    @classmethod
    def hmac_sha256(
        cls,
        secret: str | SecretStr,
        timestamp_tolerance_seconds: int = 300,
    ) -> WebhookSignatureConfig:
        """Create an HMAC-SHA256 signature configuration.

        Args:
            secret: Shared secret for HMAC computation.
            timestamp_tolerance_seconds: Max allowed timestamp age in seconds.

        Returns:
            Configured WebhookSignatureConfig for HMAC-SHA256.
        """
        if isinstance(secret, str):
            secret = SecretStr(secret)
        return cls(
            mode=WebhookSignatureMode.HMAC_SHA256,
            secret=secret,
            timestamp_tolerance_seconds=timestamp_tolerance_seconds,
        )
