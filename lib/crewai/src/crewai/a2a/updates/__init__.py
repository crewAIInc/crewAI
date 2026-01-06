"""A2A update mechanism configuration types."""

from crewai.a2a.updates.base import (
    BaseHandlerKwargs,
    PollingHandlerKwargs,
    PushNotificationHandlerKwargs,
    StreamingHandlerKwargs,
    UpdateHandler,
)
from crewai.a2a.updates.polling.config import PollingConfig
from crewai.a2a.updates.polling.handler import PollingHandler
from crewai.a2a.updates.push_notifications.config import PushNotificationConfig
from crewai.a2a.updates.push_notifications.handler import PushNotificationHandler
from crewai.a2a.updates.streaming.config import StreamingConfig
from crewai.a2a.updates.streaming.handler import StreamingHandler


UpdateConfig = PollingConfig | StreamingConfig | PushNotificationConfig

__all__ = [
    "BaseHandlerKwargs",
    "PollingConfig",
    "PollingHandler",
    "PollingHandlerKwargs",
    "PushNotificationConfig",
    "PushNotificationHandler",
    "PushNotificationHandlerKwargs",
    "StreamingConfig",
    "StreamingHandler",
    "StreamingHandlerKwargs",
    "UpdateConfig",
    "UpdateHandler",
]
