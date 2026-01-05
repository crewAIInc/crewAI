"""A2A update mechanism configuration types."""

from crewai.a2a.updates.polling.config import PollingConfig
from crewai.a2a.updates.push_notifications.config import PushNotificationConfig
from crewai.a2a.updates.streaming.config import StreamingConfig


UpdateConfig = PollingConfig | StreamingConfig | PushNotificationConfig

__all__ = [
    "PollingConfig",
    "PushNotificationConfig",
    "StreamingConfig",
    "UpdateConfig",
]
