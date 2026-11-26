"""Validation tests for webhook HMAC signature configuration."""

from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError

from crewai.a2a.updates.push_notifications.config import PushNotificationConfig
from crewai.a2a.updates.push_notifications.signature import (
    WebhookSignatureConfig,
    WebhookSignatureMode,
)


def test_hmac_mode_requires_a_secret() -> None:
    with pytest.raises(ValidationError, match="secret is required"):
        WebhookSignatureConfig(mode=WebhookSignatureMode.HMAC_SHA256)


def test_hmac_mode_rejects_an_empty_secret() -> None:
    with pytest.raises(ValidationError, match="secret is required"):
        WebhookSignatureConfig(
            mode=WebhookSignatureMode.HMAC_SHA256,
            secret=SecretStr(""),
        )


def test_none_mode_allows_a_missing_secret() -> None:
    config = WebhookSignatureConfig()

    assert config.mode is WebhookSignatureMode.NONE
    assert config.secret is None


def test_hmac_helper_keeps_a_valid_secret() -> None:
    config = WebhookSignatureConfig.hmac_sha256("shared-secret")

    assert config.mode is WebhookSignatureMode.HMAC_SHA256
    assert config.secret is not None
    assert config.secret.get_secret_value() == "shared-secret"


def test_push_notification_config_rejects_an_empty_string_secret() -> None:
    with pytest.raises(ValidationError, match="secret is required"):
        PushNotificationConfig(
            url="http://localhost:8080/a2a/callback",
            signature="",
        )
