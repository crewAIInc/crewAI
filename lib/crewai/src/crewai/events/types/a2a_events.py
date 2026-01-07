"""Events for A2A (Agent-to-Agent) delegation.

This module defines events emitted during A2A protocol delegation,
including both single-turn and multiturn conversation flows.
"""

from typing import Any, Literal

from crewai.events.base_events import BaseEvent


class A2AEventBase(BaseEvent):
    """Base class for A2A events with task/agent context."""

    from_task: Any | None = None
    from_agent: Any | None = None

    def __init__(self, **data: Any) -> None:
        """Initialize A2A event, extracting task and agent metadata."""
        if data.get("from_task"):
            task = data["from_task"]
            data["task_id"] = str(task.id)
            data["task_name"] = task.name or task.description
            data["from_task"] = None

        if data.get("from_agent"):
            agent = data["from_agent"]
            data["agent_id"] = str(agent.id)
            data["agent_role"] = agent.role
            data["from_agent"] = None

        super().__init__(**data)


class A2ADelegationStartedEvent(A2AEventBase):
    """Event emitted when A2A delegation starts.

    Attributes:
        endpoint: A2A agent endpoint URL (AgentCard URL)
        task_description: Task being delegated to the A2A agent
        agent_id: A2A agent identifier
        is_multiturn: Whether this is part of a multiturn conversation
        turn_number: Current turn number (1-indexed, 1 for single-turn)
    """

    type: str = "a2a_delegation_started"
    endpoint: str
    task_description: str
    agent_id: str
    is_multiturn: bool = False
    turn_number: int = 1


class A2ADelegationCompletedEvent(A2AEventBase):
    """Event emitted when A2A delegation completes.

    Attributes:
        status: Completion status (completed, input_required, failed, etc.)
        result: Result message if status is completed
        error: Error/response message (error for failed, response for input_required)
        is_multiturn: Whether this is part of a multiturn conversation
    """

    type: str = "a2a_delegation_completed"
    status: str
    result: str | None = None
    error: str | None = None
    is_multiturn: bool = False


class A2AConversationStartedEvent(A2AEventBase):
    """Event emitted when a multiturn A2A conversation starts.

    This is emitted once at the beginning of a multiturn conversation,
    before the first message exchange.

    Attributes:
        agent_id: A2A agent identifier
        endpoint: A2A agent endpoint URL
        a2a_agent_name: Name of the A2A agent from agent card
    """

    type: str = "a2a_conversation_started"
    agent_id: str
    endpoint: str
    a2a_agent_name: str | None = None


class A2AMessageSentEvent(A2AEventBase):
    """Event emitted when a message is sent to the A2A agent.

    Attributes:
        message: Message content sent to the A2A agent
        turn_number: Current turn number (1-indexed)
        is_multiturn: Whether this is part of a multiturn conversation
        agent_role: Role of the CrewAI agent sending the message
    """

    type: str = "a2a_message_sent"
    message: str
    turn_number: int
    is_multiturn: bool = False
    agent_role: str | None = None


class A2AResponseReceivedEvent(A2AEventBase):
    """Event emitted when a response is received from the A2A agent.

    Attributes:
        response: Response content from the A2A agent
        turn_number: Current turn number (1-indexed)
        is_multiturn: Whether this is part of a multiturn conversation
        status: Response status (input_required, completed, etc.)
        agent_role: Role of the CrewAI agent (for display)
    """

    type: str = "a2a_response_received"
    response: str
    turn_number: int
    is_multiturn: bool = False
    status: str
    agent_role: str | None = None


class A2AConversationCompletedEvent(A2AEventBase):
    """Event emitted when a multiturn A2A conversation completes.

    This is emitted once at the end of a multiturn conversation.

    Attributes:
        status: Final status (completed, failed, etc.)
        final_result: Final result if completed successfully
        error: Error message if failed
        total_turns: Total number of turns in the conversation
    """

    type: str = "a2a_conversation_completed"
    status: Literal["completed", "failed"]
    final_result: str | None = None
    error: str | None = None
    total_turns: int


class A2APollingStartedEvent(A2AEventBase):
    """Event emitted when polling mode begins for A2A delegation.

    Attributes:
        task_id: A2A task ID being polled
        polling_interval: Seconds between poll attempts
        endpoint: A2A agent endpoint URL
    """

    type: str = "a2a_polling_started"
    task_id: str
    polling_interval: float
    endpoint: str


class A2APollingStatusEvent(A2AEventBase):
    """Event emitted on each polling iteration.

    Attributes:
        task_id: A2A task ID being polled
        state: Current task state from remote agent
        elapsed_seconds: Time since polling started
        poll_count: Number of polls completed
    """

    type: str = "a2a_polling_status"
    task_id: str
    state: str
    elapsed_seconds: float
    poll_count: int


class A2APushNotificationRegisteredEvent(A2AEventBase):
    """Event emitted when push notification callback is registered.

    Attributes:
        task_id: A2A task ID for which callback is registered
        callback_url: URL where agent will send push notifications
    """

    type: str = "a2a_push_notification_registered"
    task_id: str
    callback_url: str


class A2APushNotificationReceivedEvent(A2AEventBase):
    """Event emitted when a push notification is received.

    Attributes:
        task_id: A2A task ID from the notification
        state: Current task state from the notification
    """

    type: str = "a2a_push_notification_received"
    task_id: str
    state: str


class A2APushNotificationTimeoutEvent(A2AEventBase):
    """Event emitted when push notification wait times out.

    Attributes:
        task_id: A2A task ID that timed out
        timeout_seconds: Timeout duration in seconds
    """

    type: str = "a2a_push_notification_timeout"
    task_id: str
    timeout_seconds: float
