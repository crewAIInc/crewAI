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

    def __init__(self, **data):
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
