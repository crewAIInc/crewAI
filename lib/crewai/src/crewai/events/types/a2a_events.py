"""Events for A2A (Agent-to-Agent) delegation.

This module defines events emitted during A2A protocol delegation,
including both single-turn and multiturn conversation flows.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import model_validator

from crewai.events.base_events import BaseEvent


class A2AEventBase(BaseEvent):
    """Base class for A2A events with task/agent context."""

    from_task: Any = None
    from_agent: Any = None

    @model_validator(mode="before")
    @classmethod
    def extract_task_and_agent_metadata(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Extract task and agent metadata before validation."""
        if task := data.get("from_task"):
            data["task_id"] = str(task.id)
            data["task_name"] = task.name or task.description
            data.setdefault("source_fingerprint", str(task.id))
            data.setdefault("source_type", "task")
            data.setdefault(
                "fingerprint_metadata",
                {
                    "task_id": str(task.id),
                    "task_name": task.name or task.description,
                },
            )
            data["from_task"] = None

        if agent := data.get("from_agent"):
            data["agent_id"] = str(agent.id)
            data["agent_role"] = agent.role
            data.setdefault("source_fingerprint", str(agent.id))
            data.setdefault("source_type", "agent")
            data.setdefault(
                "fingerprint_metadata",
                {
                    "agent_id": str(agent.id),
                    "agent_role": agent.role,
                },
            )
            data["from_agent"] = None

        return data


class A2ADelegationStartedEvent(A2AEventBase):
    """Event emitted when A2A delegation starts.

    Attributes:
        endpoint: A2A agent endpoint URL (AgentCard URL).
        task_description: Task being delegated to the A2A agent.
        agent_id: A2A agent identifier.
        context_id: A2A context ID grouping related tasks.
        is_multiturn: Whether this is part of a multiturn conversation.
        turn_number: Current turn number (1-indexed, 1 for single-turn).
        a2a_agent_name: Name of the A2A agent from agent card.
        agent_card: Full A2A agent card metadata.
        protocol_version: A2A protocol version being used.
        provider: Agent provider/organization info from agent card.
        skill_id: ID of the specific skill being invoked.
        metadata: Custom A2A metadata key-value pairs.
        extensions: List of A2A extension URIs in use.
    """

    type: str = "a2a_delegation_started"
    endpoint: str
    task_description: str
    agent_id: str
    context_id: str | None = None
    is_multiturn: bool = False
    turn_number: int = 1
    a2a_agent_name: str | None = None
    agent_card: dict[str, Any] | None = None
    protocol_version: str | None = None
    provider: dict[str, Any] | None = None
    skill_id: str | None = None
    metadata: dict[str, Any] | None = None
    extensions: list[str] | None = None


class A2ADelegationCompletedEvent(A2AEventBase):
    """Event emitted when A2A delegation completes.

    Attributes:
        status: Completion status (completed, input_required, failed, etc.).
        result: Result message if status is completed.
        error: Error/response message (error for failed, response for input_required).
        context_id: A2A context ID grouping related tasks.
        is_multiturn: Whether this is part of a multiturn conversation.
        endpoint: A2A agent endpoint URL.
        a2a_agent_name: Name of the A2A agent from agent card.
        agent_card: Full A2A agent card metadata.
        provider: Agent provider/organization info from agent card.
        metadata: Custom A2A metadata key-value pairs.
        extensions: List of A2A extension URIs in use.
    """

    type: str = "a2a_delegation_completed"
    status: str
    result: str | None = None
    error: str | None = None
    context_id: str | None = None
    is_multiturn: bool = False
    endpoint: str | None = None
    a2a_agent_name: str | None = None
    agent_card: dict[str, Any] | None = None
    provider: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    extensions: list[str] | None = None


class A2AConversationStartedEvent(A2AEventBase):
    """Event emitted when a multiturn A2A conversation starts.

    This is emitted once at the beginning of a multiturn conversation,
    before the first message exchange.

    Attributes:
        agent_id: A2A agent identifier.
        endpoint: A2A agent endpoint URL.
        context_id: A2A context ID grouping related tasks.
        a2a_agent_name: Name of the A2A agent from agent card.
        agent_card: Full A2A agent card metadata.
        protocol_version: A2A protocol version being used.
        provider: Agent provider/organization info from agent card.
        skill_id: ID of the specific skill being invoked.
        reference_task_ids: Related task IDs for context.
        metadata: Custom A2A metadata key-value pairs.
        extensions: List of A2A extension URIs in use.
    """

    type: str = "a2a_conversation_started"
    agent_id: str
    endpoint: str
    context_id: str | None = None
    a2a_agent_name: str | None = None
    agent_card: dict[str, Any] | None = None
    protocol_version: str | None = None
    provider: dict[str, Any] | None = None
    skill_id: str | None = None
    reference_task_ids: list[str] | None = None
    metadata: dict[str, Any] | None = None
    extensions: list[str] | None = None


class A2AMessageSentEvent(A2AEventBase):
    """Event emitted when a message is sent to the A2A agent.

    Attributes:
        message: Message content sent to the A2A agent.
        turn_number: Current turn number (1-indexed).
        context_id: A2A context ID grouping related tasks.
        message_id: Unique A2A message identifier.
        is_multiturn: Whether this is part of a multiturn conversation.
        agent_role: Role of the CrewAI agent sending the message.
        endpoint: A2A agent endpoint URL.
        a2a_agent_name: Name of the A2A agent from agent card.
        skill_id: ID of the specific skill being invoked.
        metadata: Custom A2A metadata key-value pairs.
        extensions: List of A2A extension URIs in use.
    """

    type: str = "a2a_message_sent"
    message: str
    turn_number: int
    context_id: str | None = None
    message_id: str | None = None
    is_multiturn: bool = False
    agent_role: str | None = None
    endpoint: str | None = None
    a2a_agent_name: str | None = None
    skill_id: str | None = None
    metadata: dict[str, Any] | None = None
    extensions: list[str] | None = None


class A2AResponseReceivedEvent(A2AEventBase):
    """Event emitted when a response is received from the A2A agent.

    Attributes:
        response: Response content from the A2A agent.
        turn_number: Current turn number (1-indexed).
        context_id: A2A context ID grouping related tasks.
        message_id: Unique A2A message identifier.
        is_multiturn: Whether this is part of a multiturn conversation.
        status: Response status (input_required, completed, etc.).
        final: Whether this is the final response in the stream.
        agent_role: Role of the CrewAI agent (for display).
        endpoint: A2A agent endpoint URL.
        a2a_agent_name: Name of the A2A agent from agent card.
        metadata: Custom A2A metadata key-value pairs.
        extensions: List of A2A extension URIs in use.
    """

    type: str = "a2a_response_received"
    response: str
    turn_number: int
    context_id: str | None = None
    message_id: str | None = None
    is_multiturn: bool = False
    status: str
    final: bool = False
    agent_role: str | None = None
    endpoint: str | None = None
    a2a_agent_name: str | None = None
    metadata: dict[str, Any] | None = None
    extensions: list[str] | None = None


class A2AConversationCompletedEvent(A2AEventBase):
    """Event emitted when a multiturn A2A conversation completes.

    This is emitted once at the end of a multiturn conversation.

    Attributes:
        status: Final status (completed, failed, etc.).
        final_result: Final result if completed successfully.
        error: Error message if failed.
        context_id: A2A context ID grouping related tasks.
        total_turns: Total number of turns in the conversation.
        endpoint: A2A agent endpoint URL.
        a2a_agent_name: Name of the A2A agent from agent card.
        agent_card: Full A2A agent card metadata.
        reference_task_ids: Related task IDs for context.
        metadata: Custom A2A metadata key-value pairs.
        extensions: List of A2A extension URIs in use.
    """

    type: str = "a2a_conversation_completed"
    status: Literal["completed", "failed"]
    final_result: str | None = None
    error: str | None = None
    context_id: str | None = None
    total_turns: int
    endpoint: str | None = None
    a2a_agent_name: str | None = None
    agent_card: dict[str, Any] | None = None
    reference_task_ids: list[str] | None = None
    metadata: dict[str, Any] | None = None
    extensions: list[str] | None = None


class A2APollingStartedEvent(A2AEventBase):
    """Event emitted when polling mode begins for A2A delegation.

    Attributes:
        task_id: A2A task ID being polled.
        context_id: A2A context ID grouping related tasks.
        polling_interval: Seconds between poll attempts.
        endpoint: A2A agent endpoint URL.
        a2a_agent_name: Name of the A2A agent from agent card.
        metadata: Custom A2A metadata key-value pairs.
    """

    type: str = "a2a_polling_started"
    task_id: str
    context_id: str | None = None
    polling_interval: float
    endpoint: str
    a2a_agent_name: str | None = None
    metadata: dict[str, Any] | None = None


class A2APollingStatusEvent(A2AEventBase):
    """Event emitted on each polling iteration.

    Attributes:
        task_id: A2A task ID being polled.
        context_id: A2A context ID grouping related tasks.
        state: Current task state from remote agent.
        elapsed_seconds: Time since polling started.
        poll_count: Number of polls completed.
        endpoint: A2A agent endpoint URL.
        a2a_agent_name: Name of the A2A agent from agent card.
        metadata: Custom A2A metadata key-value pairs.
    """

    type: str = "a2a_polling_status"
    task_id: str
    context_id: str | None = None
    state: str
    elapsed_seconds: float
    poll_count: int
    endpoint: str | None = None
    a2a_agent_name: str | None = None
    metadata: dict[str, Any] | None = None


class A2APushNotificationRegisteredEvent(A2AEventBase):
    """Event emitted when push notification callback is registered.

    Attributes:
        task_id: A2A task ID for which callback is registered.
        context_id: A2A context ID grouping related tasks.
        callback_url: URL where agent will send push notifications.
        endpoint: A2A agent endpoint URL.
        a2a_agent_name: Name of the A2A agent from agent card.
        metadata: Custom A2A metadata key-value pairs.
    """

    type: str = "a2a_push_notification_registered"
    task_id: str
    context_id: str | None = None
    callback_url: str
    endpoint: str | None = None
    a2a_agent_name: str | None = None
    metadata: dict[str, Any] | None = None


class A2APushNotificationReceivedEvent(A2AEventBase):
    """Event emitted when a push notification is received.

    This event should be emitted by the user's webhook handler when it receives
    a push notification from the remote A2A agent, before calling
    `result_store.store_result()`.

    Attributes:
        task_id: A2A task ID from the notification.
        context_id: A2A context ID grouping related tasks.
        state: Current task state from the notification.
        endpoint: A2A agent endpoint URL.
        a2a_agent_name: Name of the A2A agent from agent card.
        metadata: Custom A2A metadata key-value pairs.
    """

    type: str = "a2a_push_notification_received"
    task_id: str
    context_id: str | None = None
    state: str
    endpoint: str | None = None
    a2a_agent_name: str | None = None
    metadata: dict[str, Any] | None = None


class A2APushNotificationSentEvent(A2AEventBase):
    """Event emitted when a push notification is sent to a callback URL.

    Emitted by the A2A server when it sends a task status update to the
    client's registered push notification callback URL.

    Attributes:
        task_id: A2A task ID being notified.
        context_id: A2A context ID grouping related tasks.
        callback_url: URL the notification was sent to.
        state: Task state being reported.
        success: Whether the notification was successfully delivered.
        error: Error message if delivery failed.
        metadata: Custom A2A metadata key-value pairs.
    """

    type: str = "a2a_push_notification_sent"
    task_id: str
    context_id: str | None = None
    callback_url: str
    state: str
    success: bool = True
    error: str | None = None
    metadata: dict[str, Any] | None = None


class A2APushNotificationTimeoutEvent(A2AEventBase):
    """Event emitted when push notification wait times out.

    Attributes:
        task_id: A2A task ID that timed out.
        context_id: A2A context ID grouping related tasks.
        timeout_seconds: Timeout duration in seconds.
        endpoint: A2A agent endpoint URL.
        a2a_agent_name: Name of the A2A agent from agent card.
        metadata: Custom A2A metadata key-value pairs.
    """

    type: str = "a2a_push_notification_timeout"
    task_id: str
    context_id: str | None = None
    timeout_seconds: float
    endpoint: str | None = None
    a2a_agent_name: str | None = None
    metadata: dict[str, Any] | None = None


class A2AStreamingStartedEvent(A2AEventBase):
    """Event emitted when streaming mode begins for A2A delegation.

    Attributes:
        task_id: A2A task ID for the streaming session.
        context_id: A2A context ID grouping related tasks.
        endpoint: A2A agent endpoint URL.
        a2a_agent_name: Name of the A2A agent from agent card.
        turn_number: Current turn number (1-indexed).
        is_multiturn: Whether this is part of a multiturn conversation.
        agent_role: Role of the CrewAI agent.
        metadata: Custom A2A metadata key-value pairs.
        extensions: List of A2A extension URIs in use.
    """

    type: str = "a2a_streaming_started"
    task_id: str | None = None
    context_id: str | None = None
    endpoint: str
    a2a_agent_name: str | None = None
    turn_number: int = 1
    is_multiturn: bool = False
    agent_role: str | None = None
    metadata: dict[str, Any] | None = None
    extensions: list[str] | None = None


class A2AStreamingChunkEvent(A2AEventBase):
    """Event emitted when a streaming chunk is received.

    Attributes:
        task_id: A2A task ID for the streaming session.
        context_id: A2A context ID grouping related tasks.
        chunk: The text content of the chunk.
        chunk_index: Index of this chunk in the stream (0-indexed).
        final: Whether this is the final chunk in the stream.
        endpoint: A2A agent endpoint URL.
        a2a_agent_name: Name of the A2A agent from agent card.
        turn_number: Current turn number (1-indexed).
        is_multiturn: Whether this is part of a multiturn conversation.
        metadata: Custom A2A metadata key-value pairs.
        extensions: List of A2A extension URIs in use.
    """

    type: str = "a2a_streaming_chunk"
    task_id: str | None = None
    context_id: str | None = None
    chunk: str
    chunk_index: int
    final: bool = False
    endpoint: str | None = None
    a2a_agent_name: str | None = None
    turn_number: int = 1
    is_multiturn: bool = False
    metadata: dict[str, Any] | None = None
    extensions: list[str] | None = None


class A2AAgentCardFetchedEvent(A2AEventBase):
    """Event emitted when an agent card is successfully fetched.

    Attributes:
        endpoint: A2A agent endpoint URL.
        a2a_agent_name: Name of the A2A agent from agent card.
        agent_card: Full A2A agent card metadata.
        protocol_version: A2A protocol version from agent card.
        provider: Agent provider/organization info from agent card.
        cached: Whether the agent card was retrieved from cache.
        fetch_time_ms: Time taken to fetch the agent card in milliseconds.
        metadata: Custom A2A metadata key-value pairs.
    """

    type: str = "a2a_agent_card_fetched"
    endpoint: str
    a2a_agent_name: str | None = None
    agent_card: dict[str, Any] | None = None
    protocol_version: str | None = None
    provider: dict[str, Any] | None = None
    cached: bool = False
    fetch_time_ms: float | None = None
    metadata: dict[str, Any] | None = None


class A2AAuthenticationFailedEvent(A2AEventBase):
    """Event emitted when authentication to an A2A agent fails.

    Attributes:
        endpoint: A2A agent endpoint URL.
        auth_type: Type of authentication attempted (e.g., bearer, oauth2, api_key).
        error: Error message describing the failure.
        status_code: HTTP status code if applicable.
        a2a_agent_name: Name of the A2A agent if known.
        protocol_version: A2A protocol version being used.
        metadata: Custom A2A metadata key-value pairs.
    """

    type: str = "a2a_authentication_failed"
    endpoint: str
    auth_type: str | None = None
    error: str
    status_code: int | None = None
    a2a_agent_name: str | None = None
    protocol_version: str | None = None
    metadata: dict[str, Any] | None = None


class A2AArtifactReceivedEvent(A2AEventBase):
    """Event emitted when an artifact is received from a remote A2A agent.

    Attributes:
        task_id: A2A task ID the artifact belongs to.
        artifact_id: Unique identifier for the artifact.
        artifact_name: Name of the artifact.
        artifact_description: Purpose description of the artifact.
        mime_type: MIME type of the artifact content.
        size_bytes: Size of the artifact in bytes.
        append: Whether content should be appended to existing artifact.
        last_chunk: Whether this is the final chunk of the artifact.
        endpoint: A2A agent endpoint URL.
        a2a_agent_name: Name of the A2A agent from agent card.
        context_id: Context ID for correlation.
        turn_number: Current turn number (1-indexed).
        is_multiturn: Whether this is part of a multiturn conversation.
        metadata: Custom A2A metadata key-value pairs.
        extensions: List of A2A extension URIs in use.
    """

    type: str = "a2a_artifact_received"
    task_id: str
    artifact_id: str
    artifact_name: str | None = None
    artifact_description: str | None = None
    mime_type: str | None = None
    size_bytes: int | None = None
    append: bool = False
    last_chunk: bool = False
    endpoint: str | None = None
    a2a_agent_name: str | None = None
    context_id: str | None = None
    turn_number: int = 1
    is_multiturn: bool = False
    metadata: dict[str, Any] | None = None
    extensions: list[str] | None = None


class A2AConnectionErrorEvent(A2AEventBase):
    """Event emitted when a connection error occurs during A2A communication.

    Attributes:
        endpoint: A2A agent endpoint URL.
        error: Error message describing the connection failure.
        error_type: Type of error (e.g., timeout, connection_refused, dns_error).
        status_code: HTTP status code if applicable.
        a2a_agent_name: Name of the A2A agent from agent card.
        operation: The operation being attempted when error occurred.
        context_id: A2A context ID grouping related tasks.
        task_id: A2A task ID if applicable.
        metadata: Custom A2A metadata key-value pairs.
    """

    type: str = "a2a_connection_error"
    endpoint: str
    error: str
    error_type: str | None = None
    status_code: int | None = None
    a2a_agent_name: str | None = None
    operation: str | None = None
    context_id: str | None = None
    task_id: str | None = None
    metadata: dict[str, Any] | None = None


class A2AServerTaskStartedEvent(A2AEventBase):
    """Event emitted when an A2A server task execution starts.

    Attributes:
        task_id: A2A task ID for this execution.
        context_id: A2A context ID grouping related tasks.
        metadata: Custom A2A metadata key-value pairs.
    """

    type: str = "a2a_server_task_started"
    task_id: str
    context_id: str
    metadata: dict[str, Any] | None = None


class A2AServerTaskCompletedEvent(A2AEventBase):
    """Event emitted when an A2A server task execution completes.

    Attributes:
        task_id: A2A task ID for this execution.
        context_id: A2A context ID grouping related tasks.
        result: The task result.
        metadata: Custom A2A metadata key-value pairs.
    """

    type: str = "a2a_server_task_completed"
    task_id: str
    context_id: str
    result: str
    metadata: dict[str, Any] | None = None


class A2AServerTaskCanceledEvent(A2AEventBase):
    """Event emitted when an A2A server task execution is canceled.

    Attributes:
        task_id: A2A task ID for this execution.
        context_id: A2A context ID grouping related tasks.
        metadata: Custom A2A metadata key-value pairs.
    """

    type: str = "a2a_server_task_canceled"
    task_id: str
    context_id: str
    metadata: dict[str, Any] | None = None


class A2AServerTaskFailedEvent(A2AEventBase):
    """Event emitted when an A2A server task execution fails.

    Attributes:
        task_id: A2A task ID for this execution.
        context_id: A2A context ID grouping related tasks.
        error: Error message describing the failure.
        metadata: Custom A2A metadata key-value pairs.
    """

    type: str = "a2a_server_task_failed"
    task_id: str
    context_id: str
    error: str
    metadata: dict[str, Any] | None = None


class A2AParallelDelegationStartedEvent(A2AEventBase):
    """Event emitted when parallel delegation to multiple A2A agents begins.

    Attributes:
        endpoints: List of A2A agent endpoints being delegated to.
        task_description: Description of the task being delegated.
    """

    type: str = "a2a_parallel_delegation_started"
    endpoints: list[str]
    task_description: str


class A2AParallelDelegationCompletedEvent(A2AEventBase):
    """Event emitted when parallel delegation to multiple A2A agents completes.

    Attributes:
        endpoints: List of A2A agent endpoints that were delegated to.
        success_count: Number of successful delegations.
        failure_count: Number of failed delegations.
        results: Summary of results from each agent.
    """

    type: str = "a2a_parallel_delegation_completed"
    endpoints: list[str]
    success_count: int
    failure_count: int
    results: dict[str, str] | None = None


class A2ATransportNegotiatedEvent(A2AEventBase):
    """Event emitted when transport protocol is negotiated with an A2A agent.

    This event is emitted after comparing client and server transport capabilities
    to select the optimal transport protocol and endpoint URL.

    Attributes:
        endpoint: Original A2A agent endpoint URL.
        a2a_agent_name: Name of the A2A agent from agent card.
        negotiated_transport: The transport protocol selected (JSONRPC, GRPC, HTTP+JSON).
        negotiated_url: The URL to use for the selected transport.
        source: How the transport was selected ('client_preferred', 'server_preferred', 'fallback').
        client_supported_transports: Transports the client can use.
        server_supported_transports: Transports the server supports.
        server_preferred_transport: The server's preferred transport from AgentCard.
        client_preferred_transport: The client's preferred transport if set.
        metadata: Custom A2A metadata key-value pairs.
    """

    type: str = "a2a_transport_negotiated"
    endpoint: str
    a2a_agent_name: str | None = None
    negotiated_transport: str
    negotiated_url: str
    source: str
    client_supported_transports: list[str]
    server_supported_transports: list[str]
    server_preferred_transport: str
    client_preferred_transport: str | None = None
    metadata: dict[str, Any] | None = None


class A2AContentTypeNegotiatedEvent(A2AEventBase):
    """Event emitted when content types are negotiated with an A2A agent.

    This event is emitted after comparing client and server input/output mode
    capabilities to determine compatible MIME types for communication.

    Attributes:
        endpoint: A2A agent endpoint URL.
        a2a_agent_name: Name of the A2A agent from agent card.
        skill_name: Skill name if negotiation was skill-specific.
        client_input_modes: MIME types the client can send.
        client_output_modes: MIME types the client can accept.
        server_input_modes: MIME types the server accepts.
        server_output_modes: MIME types the server produces.
        negotiated_input_modes: Compatible input MIME types selected.
        negotiated_output_modes: Compatible output MIME types selected.
        negotiation_success: Whether compatible types were found for both directions.
        metadata: Custom A2A metadata key-value pairs.
    """

    type: str = "a2a_content_type_negotiated"
    endpoint: str
    a2a_agent_name: str | None = None
    skill_name: str | None = None
    client_input_modes: list[str]
    client_output_modes: list[str]
    server_input_modes: list[str]
    server_output_modes: list[str]
    negotiated_input_modes: list[str]
    negotiated_output_modes: list[str]
    negotiation_success: bool = True
    metadata: dict[str, Any] | None = None


# -----------------------------------------------------------------------------
# Context Lifecycle Events
# -----------------------------------------------------------------------------


class A2AContextCreatedEvent(A2AEventBase):
    """Event emitted when an A2A context is created.

    Contexts group related tasks in a conversation or workflow.

    Attributes:
        context_id: Unique identifier for the context.
        created_at: Unix timestamp when context was created.
        metadata: Custom A2A metadata key-value pairs.
    """

    type: str = "a2a_context_created"
    context_id: str
    created_at: float
    metadata: dict[str, Any] | None = None


class A2AContextExpiredEvent(A2AEventBase):
    """Event emitted when an A2A context expires due to TTL.

    Attributes:
        context_id: The expired context identifier.
        created_at: Unix timestamp when context was created.
        age_seconds: How long the context existed before expiring.
        task_count: Number of tasks in the context when expired.
        metadata: Custom A2A metadata key-value pairs.
    """

    type: str = "a2a_context_expired"
    context_id: str
    created_at: float
    age_seconds: float
    task_count: int
    metadata: dict[str, Any] | None = None


class A2AContextIdleEvent(A2AEventBase):
    """Event emitted when an A2A context becomes idle.

    Idle contexts have had no activity for the configured threshold.

    Attributes:
        context_id: The idle context identifier.
        idle_seconds: Seconds since last activity.
        task_count: Number of tasks in the context.
        metadata: Custom A2A metadata key-value pairs.
    """

    type: str = "a2a_context_idle"
    context_id: str
    idle_seconds: float
    task_count: int
    metadata: dict[str, Any] | None = None


class A2AContextCompletedEvent(A2AEventBase):
    """Event emitted when all tasks in an A2A context complete.

    Attributes:
        context_id: The completed context identifier.
        total_tasks: Total number of tasks that were in the context.
        duration_seconds: Total context lifetime in seconds.
        metadata: Custom A2A metadata key-value pairs.
    """

    type: str = "a2a_context_completed"
    context_id: str
    total_tasks: int
    duration_seconds: float
    metadata: dict[str, Any] | None = None


class A2AContextPrunedEvent(A2AEventBase):
    """Event emitted when an A2A context is pruned (deleted).

    Pruning removes the context metadata and optionally associated tasks.

    Attributes:
        context_id: The pruned context identifier.
        task_count: Number of tasks that were in the context.
        age_seconds: How long the context existed before pruning.
        metadata: Custom A2A metadata key-value pairs.
    """

    type: str = "a2a_context_pruned"
    context_id: str
    task_count: int
    age_seconds: float
    metadata: dict[str, Any] | None = None
