"""A2A update mechanism configuration types."""

from crewai_a2a.updates.base import (
    BaseHandlerKwargs,
    PollingHandlerKwargs,
    PushNotificationHandlerKwargs,
    PushNotificationResultStore,
    StreamingHandlerKwargs,
    UpdateHandler,
)
from crewai_a2a.updates.polling.config import PollingConfig
from crewai_a2a.updates.polling.handler import PollingHandler
from crewai_a2a.updates.push_notifications.config import PushNotificationConfig
from crewai_a2a.updates.push_notifications.handler import PushNotificationHandler
from crewai_a2a.updates.streaming.config import StreamingConfig
from crewai_a2a.updates.streaming.handler import StreamingHandler


UpdateConfig = PollingConfig | StreamingConfig | PushNotificationConfig

__all__ = [
    "BaseHandlerKwargs",
    "PollingConfig",
    "PollingHandler",
    "PollingHandlerKwargs",
    "PushNotificationConfig",
    "PushNotificationHandler",
    "PushNotificationHandlerKwargs",
    "PushNotificationResultStore",
    "StreamingConfig",
    "StreamingHandler",
    "StreamingHandlerKwargs",
    "UpdateConfig",
    "UpdateHandler",
]
