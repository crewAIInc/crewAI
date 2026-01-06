"""Push notification (webhook) update mechanism handler."""

from __future__ import annotations

from typing import Unpack

from a2a.client import Client
from a2a.types import AgentCard, Message

from crewai.a2a.task_helpers import TaskStateResult
from crewai.a2a.updates.base import PushNotificationHandlerKwargs


class PushNotificationHandler:
    """Push notification (webhook) based update handler."""

    @staticmethod
    async def execute(
        client: Client,
        message: Message,
        new_messages: list[Message],
        agent_card: AgentCard,
        **kwargs: Unpack[PushNotificationHandlerKwargs],
    ) -> TaskStateResult:
        """Execute A2A delegation using push notifications for updates.

        Args:
            client: A2A client instance.
            message: Message to send.
            new_messages: List to collect messages.
            agent_card: The agent card.
            **kwargs: Push notification-specific parameters.

        Raises:
            NotImplementedError: Push notifications not yet implemented.
        """
        raise NotImplementedError(
            "Push notification update mechanism is not yet implemented. "
            "Use PollingConfig or StreamingConfig instead."
        )
