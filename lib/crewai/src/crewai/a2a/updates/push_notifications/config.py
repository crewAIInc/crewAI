"""Push notification update mechanism configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field

from crewai.a2a.auth.schemas import AuthScheme


class PushNotificationConfig(BaseModel):
    """Configuration for webhook-based task updates.

    Attributes:
        url: Callback URL where agent sends push notifications.
        id: Unique identifier for this config.
        token: Token to validate incoming notifications.
        authentication: Auth scheme for the callback endpoint.
    """

    url: str = Field(description="Callback URL for push notifications")
    id: str | None = Field(default=None, description="Unique config identifier")
    token: str | None = Field(default=None, description="Validation token")
    authentication: AuthScheme | None = Field(
        default=None, description="Authentication for callback endpoint"
    )
