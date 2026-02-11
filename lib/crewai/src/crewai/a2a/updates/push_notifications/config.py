"""Push notification update mechanism configuration."""

from __future__ import annotations

from typing import Annotated

from a2a.types import PushNotificationAuthenticationInfo
from pydantic import AnyHttpUrl, BaseModel, BeforeValidator, Field

from crewai.a2a.updates.base import PushNotificationResultStore
from crewai.a2a.updates.push_notifications.signature import WebhookSignatureConfig


def _coerce_signature(
    value: str | WebhookSignatureConfig | None,
) -> WebhookSignatureConfig | None:
    """Convert string secret to WebhookSignatureConfig."""
    if value is None:
        return None
    if isinstance(value, str):
        return WebhookSignatureConfig.hmac_sha256(secret=value)
    return value


SignatureInput = Annotated[
    WebhookSignatureConfig | None,
    BeforeValidator(_coerce_signature),
]


class PushNotificationConfig(BaseModel):
    """Configuration for webhook-based task updates.

    Attributes:
        url: Callback URL where agent sends push notifications.
        id: Unique identifier for this config.
        token: Token to validate incoming notifications.
        authentication: Auth info for agent to use when calling webhook.
        timeout: Max seconds to wait for task completion.
        interval: Seconds between result polling attempts.
        result_store: Store for receiving push notification results.
        signature: HMAC signature config. Pass a string (secret) for defaults,
            or WebhookSignatureConfig for custom settings.
    """

    url: AnyHttpUrl = Field(description="Callback URL for push notifications")
    id: str | None = Field(default=None, description="Unique config identifier")
    token: str | None = Field(default=None, description="Validation token")
    authentication: PushNotificationAuthenticationInfo | None = Field(
        default=None, description="Auth info for agent to use when calling webhook"
    )
    timeout: float | None = Field(
        default=300.0, gt=0, description="Max seconds to wait for task completion"
    )
    interval: float = Field(
        default=2.0, gt=0, description="Seconds between result polling attempts"
    )
    result_store: PushNotificationResultStore | None = Field(
        default=None, description="Result store for push notification handling"
    )
    signature: SignatureInput = Field(
        default=None,
        description="HMAC signature config. Pass a string (secret) for simple usage, "
        "or WebhookSignatureConfig for custom headers/tolerance.",
    )
