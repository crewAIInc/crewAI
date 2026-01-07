"""Push notification update mechanism configuration."""

from __future__ import annotations

from a2a.types import PushNotificationAuthenticationInfo
from pydantic import AnyHttpUrl, BaseModel, Field

from crewai.a2a.updates.base import PushNotificationResultStore


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
