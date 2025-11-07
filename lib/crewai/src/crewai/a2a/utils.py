"""Utility functions for A2A (Agent-to-Agent) protocol delegation."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, MutableMapping
from contextlib import asynccontextmanager
from functools import lru_cache
import time
from typing import TYPE_CHECKING, Any
import uuid

from a2a.client import Client, ClientConfig, ClientFactory
from a2a.client.errors import A2AClientHTTPError
from a2a.types import (
    AgentCard,
    Message,
    Part,
    Role,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatusUpdateEvent,
    TextPart,
    TransportProtocol,
)
import httpx
from pydantic import BaseModel, Field, create_model

from crewai.a2a.auth.schemas import APIKeyAuth, HTTPDigestAuth
from crewai.a2a.auth.utils import (
    _auth_store,
    configure_auth_client,
    retry_on_401,
    validate_auth_against_agent_card,
)
from crewai.a2a.config import A2AConfig
from crewai.a2a.types import PartsDict, PartsMetadataDict
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.a2a_events import (
    A2AConversationStartedEvent,
    A2ADelegationCompletedEvent,
    A2ADelegationStartedEvent,
    A2AMessageSentEvent,
    A2AResponseReceivedEvent,
)
from crewai.types.utils import create_literals_from_strings


if TYPE_CHECKING:
    from a2a.types import Message, Task as A2ATask

    from crewai.a2a.auth.schemas import AuthScheme


@lru_cache()
def _fetch_agent_card_cached(
    endpoint: str,
    auth_hash: int,
    timeout: int,
    _ttl_hash: int,
) -> AgentCard:
    """Cached version of fetch_agent_card with auth support.

    Args:
        endpoint: A2A agent endpoint URL
        auth_hash: Hash of the auth object
        timeout: Request timeout
        _ttl_hash: Time-based hash for cache invalidation (unused in body)

    Returns:
        Cached AgentCard
    """
    auth = _auth_store.get(auth_hash)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            _fetch_agent_card_async(endpoint=endpoint, auth=auth, timeout=timeout)
        )
    finally:
        loop.close()


def fetch_agent_card(
    endpoint: str,
    auth: AuthScheme | None = None,
    timeout: int = 30,
    use_cache: bool = True,
    cache_ttl: int = 300,
) -> AgentCard:
    """Fetch AgentCard from an A2A endpoint with optional caching.

    Args:
        endpoint: A2A agent endpoint URL (AgentCard URL)
        auth: Optional AuthScheme for authentication
        timeout: Request timeout in seconds
        use_cache: Whether to use caching (default True)
        cache_ttl: Cache TTL in seconds (default 300 = 5 minutes)

    Returns:
        AgentCard object with agent capabilities and skills

    Raises:
        httpx.HTTPStatusError: If the request fails
        A2AClientHTTPError: If authentication fails
    """
    if use_cache:
        auth_hash = hash((type(auth).__name__, id(auth))) if auth else 0
        _auth_store[auth_hash] = auth
        ttl_hash = int(time.time() // cache_ttl)
        return _fetch_agent_card_cached(endpoint, auth_hash, timeout, ttl_hash)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            _fetch_agent_card_async(endpoint=endpoint, auth=auth, timeout=timeout)
        )
    finally:
        loop.close()


async def _fetch_agent_card_async(
    endpoint: str,
    auth: AuthScheme | None,
    timeout: int,
) -> AgentCard:
    """Async implementation of AgentCard fetching.

    Args:
        endpoint: A2A agent endpoint URL
        auth: Optional AuthScheme for authentication
        timeout: Request timeout in seconds

    Returns:
        AgentCard object
    """
    if "/.well-known/agent-card.json" in endpoint:
        base_url = endpoint.replace("/.well-known/agent-card.json", "")
        agent_card_path = "/.well-known/agent-card.json"
    else:
        url_parts = endpoint.split("/", 3)
        base_url = f"{url_parts[0]}//{url_parts[2]}"
        agent_card_path = f"/{url_parts[3]}" if len(url_parts) > 3 else "/"

    headers: MutableMapping[str, str] = {}
    if auth:
        async with httpx.AsyncClient(timeout=timeout) as temp_auth_client:
            if isinstance(auth, (HTTPDigestAuth, APIKeyAuth)):
                configure_auth_client(auth, temp_auth_client)
            headers = await auth.apply_auth(temp_auth_client, {})

    async with httpx.AsyncClient(timeout=timeout, headers=headers) as temp_client:
        if auth and isinstance(auth, (HTTPDigestAuth, APIKeyAuth)):
            configure_auth_client(auth, temp_client)

        agent_card_url = f"{base_url}{agent_card_path}"

        async def _fetch_agent_card_request() -> httpx.Response:
            return await temp_client.get(agent_card_url)

        try:
            response = await retry_on_401(
                request_func=_fetch_agent_card_request,
                auth_scheme=auth,
                client=temp_client,
                headers=temp_client.headers,
                max_retries=2,
            )
            response.raise_for_status()

            return AgentCard.model_validate(response.json())

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                error_details = ["Authentication failed"]
                www_auth = e.response.headers.get("WWW-Authenticate")
                if www_auth:
                    error_details.append(f"WWW-Authenticate: {www_auth}")
                if not auth:
                    error_details.append("No auth scheme provided")
                msg = " | ".join(error_details)
                raise A2AClientHTTPError(401, msg) from e
            raise


def execute_a2a_delegation(
    endpoint: str,
    auth: AuthScheme | None,
    timeout: int,
    task_description: str,
    context: str | None = None,
    context_id: str | None = None,
    task_id: str | None = None,
    reference_task_ids: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    extensions: dict[str, Any] | None = None,
    conversation_history: list[Message] | None = None,
    agent_id: str | None = None,
    agent_role: Role | None = None,
    agent_branch: Any | None = None,
    response_model: type[BaseModel] | None = None,
    turn_number: int | None = None,
) -> dict[str, Any]:
    """Execute a task delegation to a remote A2A agent with multi-turn support.

    Handles:
    - AgentCard discovery
    - Authentication setup
    - Message creation and sending
    - Response parsing
    - Multi-turn conversations

    Args:
        endpoint: A2A agent endpoint URL (AgentCard URL)
        auth: Optional AuthScheme for authentication (Bearer, OAuth2, API Key, HTTP Basic/Digest)
        timeout: Request timeout in seconds
        task_description: The task to delegate
        context: Optional context information
        context_id: Context ID for correlating messages/tasks
        task_id: Specific task identifier
        reference_task_ids: List of related task IDs
        metadata: Additional metadata (external_id, request_id, etc.)
        extensions: Protocol extensions for custom fields
        conversation_history: Previous Message objects from conversation
        agent_id: Agent identifier for logging
        agent_role: Role of the CrewAI agent delegating the task
        agent_branch: Optional agent tree branch for logging
        response_model: Optional Pydantic model for structured outputs
        turn_number: Optional turn number for multi-turn conversations

    Returns:
        Dictionary with:
        - status: "completed", "input_required", "failed", etc.
        - result: Result string (if completed)
        - error: Error message (if failed)
        - history: List of new Message objects from this exchange

    Raises:
        ImportError: If a2a-sdk is not installed
    """
    is_multiturn = bool(conversation_history and len(conversation_history) > 0)
    if turn_number is None:
        turn_number = (
            len([m for m in (conversation_history or []) if m.role == Role.user]) + 1
        )
    crewai_event_bus.emit(
        agent_branch,
        A2ADelegationStartedEvent(
            endpoint=endpoint,
            task_description=task_description,
            agent_id=agent_id,
            is_multiturn=is_multiturn,
            turn_number=turn_number,
        ),
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(
            _execute_a2a_delegation_async(
                endpoint=endpoint,
                auth=auth,
                timeout=timeout,
                task_description=task_description,
                context=context,
                context_id=context_id,
                task_id=task_id,
                reference_task_ids=reference_task_ids,
                metadata=metadata,
                extensions=extensions,
                conversation_history=conversation_history or [],
                is_multiturn=is_multiturn,
                turn_number=turn_number,
                agent_branch=agent_branch,
                agent_id=agent_id,
                agent_role=agent_role,
                response_model=response_model,
            )
        )

        crewai_event_bus.emit(
            agent_branch,
            A2ADelegationCompletedEvent(
                status=result["status"],
                result=result.get("result"),
                error=result.get("error"),
                is_multiturn=is_multiturn,
            ),
        )

        return result
    finally:
        loop.close()


async def _execute_a2a_delegation_async(
    endpoint: str,
    auth: AuthScheme | None,
    timeout: int,
    task_description: str,
    context: str | None,
    context_id: str | None,
    task_id: str | None,
    reference_task_ids: list[str] | None,
    metadata: dict[str, Any] | None,
    extensions: dict[str, Any] | None,
    conversation_history: list[Message],
    is_multiturn: bool = False,
    turn_number: int = 1,
    agent_branch: Any | None = None,
    agent_id: str | None = None,
    agent_role: str | None = None,
    response_model: type[BaseModel] | None = None,
) -> dict[str, Any]:
    """Async implementation of A2A delegation with multi-turn support.

    Args:
        endpoint: A2A agent endpoint URL
        auth: Optional AuthScheme for authentication
        timeout: Request timeout in seconds
        task_description: Task to delegate
        context: Optional context
        context_id: Context ID for correlation
        task_id: Specific task identifier
        reference_task_ids: Related task IDs
        metadata: Additional metadata
        extensions: Protocol extensions
        conversation_history: Previous Message objects
        is_multiturn: Whether this is a multi-turn conversation
        turn_number: Current turn number
        agent_branch: Agent tree branch for logging
        agent_id: Agent identifier for logging
        agent_role: Agent role for logging
        response_model: Optional Pydantic model for structured outputs

    Returns:
        Dictionary with status, result/error, and new history
    """
    agent_card = await _fetch_agent_card_async(endpoint, auth, timeout)

    validate_auth_against_agent_card(agent_card, auth)

    headers: MutableMapping[str, str] = {}
    if auth:
        async with httpx.AsyncClient(timeout=timeout) as temp_auth_client:
            if isinstance(auth, (HTTPDigestAuth, APIKeyAuth)):
                configure_auth_client(auth, temp_auth_client)
            headers = await auth.apply_auth(temp_auth_client, {})

    a2a_agent_name = None
    if agent_card.name:
        a2a_agent_name = agent_card.name

    if turn_number == 1:
        agent_id_for_event = agent_id or endpoint
        crewai_event_bus.emit(
            agent_branch,
            A2AConversationStartedEvent(
                agent_id=agent_id_for_event,
                endpoint=endpoint,
                a2a_agent_name=a2a_agent_name,
            ),
        )

    message_parts = []

    if context:
        message_parts.append(f"Context:\n{context}\n\n")
    message_parts.append(f"{task_description}")
    message_text = "".join(message_parts)

    if is_multiturn and conversation_history and not task_id:
        if first_task_id := conversation_history[0].task_id:
            task_id = first_task_id

    parts: PartsDict = {"text": message_text}
    if response_model:
        parts.update(
            {
                "metadata": PartsMetadataDict(
                    mimeType="application/json",
                    schema=response_model.model_json_schema(),
                )
            }
        )

    message = Message(
        role=Role.user,
        message_id=str(uuid.uuid4()),
        parts=[Part(root=TextPart(**parts))],
        context_id=context_id,
        task_id=task_id,
        reference_task_ids=reference_task_ids,
        metadata=metadata,
        extensions=extensions,
    )

    transport_protocol = TransportProtocol("JSONRPC")
    new_messages: list[Message] = [*conversation_history, message]
    crewai_event_bus.emit(
        None,
        A2AMessageSentEvent(
            message=message_text,
            turn_number=turn_number,
            is_multiturn=is_multiturn,
            agent_role=agent_role,
        ),
    )

    async with _create_a2a_client(
        agent_card=agent_card,
        transport_protocol=transport_protocol,
        timeout=timeout,
        headers=headers,
        streaming=True,
        auth=auth,
    ) as client:
        result_parts: list[str] = []
        final_result: dict[str, Any] | None = None
        event_stream = client.send_message(message)

        try:
            async for event in event_stream:
                if isinstance(event, Message):
                    new_messages.append(event)
                    for part in event.parts:
                        if part.root.kind == "text":
                            text = part.root.text
                            result_parts.append(text)

                elif isinstance(event, tuple):
                    a2a_task, update = event

                    if isinstance(update, TaskArtifactUpdateEvent):
                        artifact = update.artifact
                        result_parts.extend(
                            part.root.text
                            for part in artifact.parts
                            if part.root.kind == "text"
                        )

                    is_final_update = False
                    if isinstance(update, TaskStatusUpdateEvent):
                        is_final_update = update.final

                    if not is_final_update and a2a_task.status.state not in [
                        TaskState.completed,
                        TaskState.input_required,
                        TaskState.failed,
                        TaskState.rejected,
                        TaskState.auth_required,
                        TaskState.canceled,
                    ]:
                        continue

                    if a2a_task.status.state == TaskState.completed:
                        extracted_parts = _extract_task_result_parts(a2a_task)
                        result_parts.extend(extracted_parts)
                        if a2a_task.history:
                            new_messages.extend(a2a_task.history)

                        response_text = " ".join(result_parts) if result_parts else ""
                        crewai_event_bus.emit(
                            None,
                            A2AResponseReceivedEvent(
                                response=response_text,
                                turn_number=turn_number,
                                is_multiturn=is_multiturn,
                                status="completed",
                                agent_role=agent_role,
                            ),
                        )

                        final_result = {
                            "status": "completed",
                            "result": response_text,
                            "history": new_messages,
                            "agent_card": agent_card,
                        }
                        break

                    if a2a_task.status.state == TaskState.input_required:
                        if a2a_task.history:
                            new_messages.extend(a2a_task.history)

                        response_text = _extract_error_message(
                            a2a_task, "Additional input required"
                        )
                        if response_text and not a2a_task.history:
                            agent_message = Message(
                                role=Role.agent,
                                message_id=str(uuid.uuid4()),
                                parts=[Part(root=TextPart(text=response_text))],
                                context_id=a2a_task.context_id
                                if hasattr(a2a_task, "context_id")
                                else None,
                                task_id=a2a_task.task_id
                                if hasattr(a2a_task, "task_id")
                                else None,
                            )
                            new_messages.append(agent_message)
                        crewai_event_bus.emit(
                            None,
                            A2AResponseReceivedEvent(
                                response=response_text,
                                turn_number=turn_number,
                                is_multiturn=is_multiturn,
                                status="input_required",
                                agent_role=agent_role,
                            ),
                        )

                        final_result = {
                            "status": "input_required",
                            "error": response_text,
                            "history": new_messages,
                            "agent_card": agent_card,
                        }
                        break

                    if a2a_task.status.state in [TaskState.failed, TaskState.rejected]:
                        error_msg = _extract_error_message(
                            a2a_task, "Task failed without error message"
                        )
                        if a2a_task.history:
                            new_messages.extend(a2a_task.history)
                        final_result = {
                            "status": "failed",
                            "error": error_msg,
                            "history": new_messages,
                        }
                        break

                    if a2a_task.status.state == TaskState.auth_required:
                        error_msg = _extract_error_message(
                            a2a_task, "Authentication required"
                        )
                        final_result = {
                            "status": "auth_required",
                            "error": error_msg,
                            "history": new_messages,
                        }
                        break

                    if a2a_task.status.state == TaskState.canceled:
                        error_msg = _extract_error_message(
                            a2a_task, "Task was canceled"
                        )
                        final_result = {
                            "status": "canceled",
                            "error": error_msg,
                            "history": new_messages,
                        }
                        break
        except Exception as e:
            current_exception: Exception | BaseException | None = e
            while current_exception:
                if hasattr(current_exception, "response"):
                    response = current_exception.response
                    if hasattr(response, "text"):
                        break
                if current_exception and hasattr(current_exception, "__cause__"):
                    current_exception = current_exception.__cause__
            raise
        finally:
            if hasattr(event_stream, "aclose"):
                await event_stream.aclose()

    if final_result:
        return final_result

    return {
        "status": "completed",
        "result": " ".join(result_parts) if result_parts else "",
        "history": new_messages,
    }


@asynccontextmanager
async def _create_a2a_client(
    agent_card: AgentCard,
    transport_protocol: TransportProtocol,
    timeout: int,
    headers: MutableMapping[str, str],
    streaming: bool,
    auth: AuthScheme | None = None,
) -> AsyncIterator[Client]:
    """Create and configure an A2A client.

    Args:
        agent_card: The A2A agent card
        transport_protocol: Transport protocol to use
        timeout: Request timeout in seconds
        headers: HTTP headers (already with auth applied)
        streaming: Enable streaming responses
        auth: Optional AuthScheme for client configuration

    Yields:
        Configured A2A client instance
    """

    async with httpx.AsyncClient(
        timeout=timeout,
        headers=headers,
    ) as httpx_client:
        if auth and isinstance(auth, (HTTPDigestAuth, APIKeyAuth)):
            configure_auth_client(auth, httpx_client)

        config = ClientConfig(
            httpx_client=httpx_client,
            supported_transports=[str(transport_protocol.value)],
            streaming=streaming,
            accepted_output_modes=["application/json"],
        )

        factory = ClientFactory(config)
        client = factory.create(agent_card)
        yield client


def _extract_task_result_parts(a2a_task: A2ATask) -> list[str]:
    """Extract result parts from A2A task history and artifacts.

    Args:
        a2a_task: A2A Task object with history and artifacts

    Returns:
        List of result text parts
    """

    result_parts: list[str] = []

    if a2a_task.history:
        for history_msg in reversed(a2a_task.history):
            if history_msg.role == Role.agent:
                result_parts.extend(
                    part.root.text
                    for part in history_msg.parts
                    if part.root.kind == "text"
                )
                break

    if a2a_task.artifacts:
        result_parts.extend(
            part.root.text
            for artifact in a2a_task.artifacts
            for part in artifact.parts
            if part.root.kind == "text"
        )

    return result_parts


def _extract_error_message(a2a_task: A2ATask, default: str) -> str:
    """Extract error message from A2A task.

    Args:
        a2a_task: A2A Task object
        default: Default message if no error found

    Returns:
        Error message string
    """
    if a2a_task.status and a2a_task.status.message:
        msg = a2a_task.status.message
        if msg:
            for part in msg.parts:
                if part.root.kind == "text":
                    return str(part.root.text)
        return str(msg)

    if a2a_task.history:
        for history_msg in reversed(a2a_task.history):
            for part in history_msg.parts:
                if part.root.kind == "text":
                    return str(part.root.text)

    return default


def create_agent_response_model(agent_ids: tuple[str, ...]) -> type[BaseModel]:
    """Create a dynamic AgentResponse model with Literal types for agent IDs.

    Args:
        agent_ids: List of available A2A agent IDs

    Returns:
        Dynamically created Pydantic model with Literal-constrained a2a_ids field
    """

    DynamicLiteral = create_literals_from_strings(agent_ids)  # noqa: N806

    return create_model(
        "AgentResponse",
        a2a_ids=(
            tuple[DynamicLiteral, ...],  # type: ignore[valid-type]
            Field(
                default_factory=tuple,
                max_length=len(agent_ids),
                description="A2A agent IDs to delegate to.",
            ),
        ),
        message=(
            str,
            Field(
                description="The message content. If is_a2a=true, this is sent to the A2A agent. If is_a2a=false, this is your final answer ending the conversation."
            ),
        ),
        is_a2a=(
            bool,
            Field(
                description="Set to true to continue the conversation by sending this message to the A2A agent and awaiting their response. Set to false ONLY when you are completely done and providing your final answer (not when asking questions)."
            ),
        ),
        __base__=BaseModel,
    )


def extract_a2a_agent_ids_from_config(
    a2a_config: list[A2AConfig] | A2AConfig | None,
) -> tuple[list[A2AConfig], tuple[str, ...]]:
    """Extract A2A agent IDs from A2A configuration.

    Args:
        a2a_config: A2A configuration

    Returns:
        List of A2A agent IDs
    """
    if a2a_config is None:
        return [], ()

    if isinstance(a2a_config, A2AConfig):
        a2a_agents = [a2a_config]
    else:
        a2a_agents = a2a_config
    return a2a_agents, tuple(config.endpoint for config in a2a_agents)


def get_a2a_agents_and_response_model(
    a2a_config: list[A2AConfig] | A2AConfig | None,
) -> tuple[list[A2AConfig], type[BaseModel]]:
    """Get A2A agent IDs and response model.

    Args:
        a2a_config: A2A configuration

    Returns:
        Tuple of A2A agent IDs and response model
    """
    a2a_agents, agent_ids = extract_a2a_agent_ids_from_config(a2a_config=a2a_config)
    return a2a_agents, create_agent_response_model(agent_ids)
