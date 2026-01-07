"""Utility functions for A2A (Agent-to-Agent) protocol delegation."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, MutableMapping
from contextlib import asynccontextmanager
from functools import lru_cache
import time
from typing import TYPE_CHECKING, Any
import uuid

from a2a.client import A2AClientHTTPError, Client, ClientConfig, ClientFactory
from a2a.types import (
    AgentCard,
    Message,
    Part,
    PushNotificationConfig as A2APushNotificationConfig,
    Role,
    TextPart,
    TransportProtocol,
)
from aiocache import cached  # type: ignore[import-untyped]
from aiocache.serializers import PickleSerializer  # type: ignore[import-untyped]
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
from crewai.a2a.task_helpers import TaskStateResult
from crewai.a2a.types import (
    HANDLER_REGISTRY,
    HandlerType,
    PartsDict,
    PartsMetadataDict,
)
from crewai.a2a.updates import (
    PollingConfig,
    PushNotificationConfig,
    StreamingHandler,
    UpdateConfig,
)
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.a2a_events import (
    A2AConversationStartedEvent,
    A2ADelegationCompletedEvent,
    A2ADelegationStartedEvent,
    A2AMessageSentEvent,
)
from crewai.types.utils import create_literals_from_strings


if TYPE_CHECKING:
    from a2a.types import Message

    from crewai.a2a.auth.schemas import AuthScheme


def get_handler(config: UpdateConfig | None) -> HandlerType:
    """Get the handler class for a given update config.

    Args:
        config: Update mechanism configuration.

    Returns:
        Handler class for the config type, defaults to StreamingHandler.
    """
    if config is None:
        return StreamingHandler
    return HANDLER_REGISTRY.get(type(config), StreamingHandler)


@lru_cache()
def _fetch_agent_card_cached(
    endpoint: str,
    auth_hash: int,
    timeout: int,
    _ttl_hash: int,
) -> AgentCard:
    """Cached sync version of fetch_agent_card."""
    auth = _auth_store.get(auth_hash)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            _afetch_agent_card_impl(endpoint=endpoint, auth=auth, timeout=timeout)
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
        if auth:
            auth_data = auth.model_dump_json(
                exclude={
                    "_access_token",
                    "_token_expires_at",
                    "_refresh_token",
                    "_authorization_callback",
                }
            )
            auth_hash = hash((type(auth).__name__, auth_data))
        else:
            auth_hash = 0
        _auth_store[auth_hash] = auth
        ttl_hash = int(time.time() // cache_ttl)
        return _fetch_agent_card_cached(endpoint, auth_hash, timeout, ttl_hash)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            afetch_agent_card(endpoint=endpoint, auth=auth, timeout=timeout)
        )
    finally:
        loop.close()


async def afetch_agent_card(
    endpoint: str,
    auth: AuthScheme | None = None,
    timeout: int = 30,
    use_cache: bool = True,
) -> AgentCard:
    """Fetch AgentCard from an A2A endpoint asynchronously.

    Native async implementation. Use this when running in an async context.

    Args:
        endpoint: A2A agent endpoint URL (AgentCard URL).
        auth: Optional AuthScheme for authentication.
        timeout: Request timeout in seconds.
        use_cache: Whether to use caching (default True).

    Returns:
        AgentCard object with agent capabilities and skills.

    Raises:
        httpx.HTTPStatusError: If the request fails.
        A2AClientHTTPError: If authentication fails.
    """
    if use_cache:
        if auth:
            auth_data = auth.model_dump_json(
                exclude={
                    "_access_token",
                    "_token_expires_at",
                    "_refresh_token",
                    "_authorization_callback",
                }
            )
            auth_hash = hash((type(auth).__name__, auth_data))
        else:
            auth_hash = 0
        _auth_store[auth_hash] = auth
        agent_card: AgentCard = await _afetch_agent_card_cached(
            endpoint, auth_hash, timeout
        )
        return agent_card

    return await _afetch_agent_card_impl(endpoint=endpoint, auth=auth, timeout=timeout)


@cached(ttl=300, serializer=PickleSerializer())  # type: ignore[untyped-decorator]
async def _afetch_agent_card_cached(
    endpoint: str,
    auth_hash: int,
    timeout: int,
) -> AgentCard:
    """Cached async implementation of AgentCard fetching."""
    auth = _auth_store.get(auth_hash)
    return await _afetch_agent_card_impl(endpoint=endpoint, auth=auth, timeout=timeout)


async def _afetch_agent_card_impl(
    endpoint: str,
    auth: AuthScheme | None,
    timeout: int,
) -> AgentCard:
    """Internal async implementation of AgentCard fetching."""
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
    updates: UpdateConfig | None = None,
) -> TaskStateResult:
    """Execute a task delegation to a remote A2A agent synchronously.

    This is the sync wrapper around aexecute_a2a_delegation. For async contexts,
    use aexecute_a2a_delegation directly.

    Args:
        endpoint: A2A agent endpoint URL.
        auth: Optional AuthScheme for authentication.
        timeout: Request timeout in seconds.
        task_description: The task to delegate.
        context: Optional context information.
        context_id: Context ID for correlating messages/tasks.
        task_id: Specific task identifier.
        reference_task_ids: List of related task IDs.
        metadata: Additional metadata.
        extensions: Protocol extensions for custom fields.
        conversation_history: Previous Message objects from conversation.
        agent_id: Agent identifier for logging.
        agent_role: Role of the CrewAI agent delegating the task.
        agent_branch: Optional agent tree branch for logging.
        response_model: Optional Pydantic model for structured outputs.
        turn_number: Optional turn number for multi-turn conversations.
        updates: Update mechanism config from A2AConfig.updates.

    Returns:
        TaskStateResult with status, result/error, history, and agent_card.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            aexecute_a2a_delegation(
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
                conversation_history=conversation_history,
                agent_id=agent_id,
                agent_role=agent_role,
                agent_branch=agent_branch,
                response_model=response_model,
                turn_number=turn_number,
                updates=updates,
            )
        )
    finally:
        loop.close()


async def aexecute_a2a_delegation(
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
    updates: UpdateConfig | None = None,
) -> TaskStateResult:
    """Execute a task delegation to a remote A2A agent asynchronously.

    Native async implementation with multi-turn support. Use this when running
    in an async context (e.g., with Crew.akickoff() or agent.aexecute_task()).

    Args:
        endpoint: A2A agent endpoint URL.
        auth: Optional AuthScheme for authentication.
        timeout: Request timeout in seconds.
        task_description: The task to delegate.
        context: Optional context information.
        context_id: Context ID for correlating messages/tasks.
        task_id: Specific task identifier.
        reference_task_ids: List of related task IDs.
        metadata: Additional metadata.
        extensions: Protocol extensions for custom fields.
        conversation_history: Previous Message objects from conversation.
        agent_id: Agent identifier for logging.
        agent_role: Role of the CrewAI agent delegating the task.
        agent_branch: Optional agent tree branch for logging.
        response_model: Optional Pydantic model for structured outputs.
        turn_number: Optional turn number for multi-turn conversations.
        updates: Update mechanism config from A2AConfig.updates.

    Returns:
        TaskStateResult with status, result/error, history, and agent_card.
    """
    if conversation_history is None:
        conversation_history = []

    is_multiturn = len(conversation_history) > 0
    if turn_number is None:
        turn_number = len([m for m in conversation_history if m.role == Role.user]) + 1

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

    result = await _aexecute_a2a_delegation_impl(
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
        conversation_history=conversation_history,
        is_multiturn=is_multiturn,
        turn_number=turn_number,
        agent_branch=agent_branch,
        agent_id=agent_id,
        agent_role=agent_role,
        response_model=response_model,
        updates=updates,
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


async def _aexecute_a2a_delegation_impl(
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
    is_multiturn: bool,
    turn_number: int,
    agent_branch: Any | None,
    agent_id: str | None,
    agent_role: str | None,
    response_model: type[BaseModel] | None,
    updates: UpdateConfig | None,
) -> TaskStateResult:
    """Internal async implementation of A2A delegation."""
    if auth:
        auth_data = auth.model_dump_json(
            exclude={
                "_access_token",
                "_token_expires_at",
                "_refresh_token",
                "_authorization_callback",
            }
        )
        auth_hash = hash((type(auth).__name__, auth_data))
    else:
        auth_hash = 0
    _auth_store[auth_hash] = auth
    agent_card = await _afetch_agent_card_cached(
        endpoint=endpoint, auth_hash=auth_hash, timeout=timeout
    )

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

    handler = get_handler(updates)
    use_polling = isinstance(updates, PollingConfig)

    handler_kwargs: dict[str, Any] = {
        "turn_number": turn_number,
        "is_multiturn": is_multiturn,
        "agent_role": agent_role,
        "context_id": context_id,
        "task_id": task_id,
        "endpoint": endpoint,
        "agent_branch": agent_branch,
    }

    if isinstance(updates, PollingConfig):
        handler_kwargs.update(
            {
                "polling_interval": updates.interval,
                "polling_timeout": updates.timeout or float(timeout),
                "history_length": updates.history_length,
                "max_polls": updates.max_polls,
            }
        )
    elif isinstance(updates, PushNotificationConfig):
        handler_kwargs.update(
            {
                "config": updates,
                "result_store": updates.result_store,
                "polling_timeout": updates.timeout or float(timeout),
                "polling_interval": updates.interval,
            }
        )

    push_config_for_client = (
        updates if isinstance(updates, PushNotificationConfig) else None
    )

    use_streaming = not use_polling and push_config_for_client is None

    async with _create_a2a_client(
        agent_card=agent_card,
        transport_protocol=transport_protocol,
        timeout=timeout,
        headers=headers,
        streaming=use_streaming,
        auth=auth,
        use_polling=use_polling,
        push_notification_config=push_config_for_client,
    ) as client:
        return await handler.execute(
            client=client,
            message=message,
            new_messages=new_messages,
            agent_card=agent_card,
            **handler_kwargs,
        )


@asynccontextmanager
async def _create_a2a_client(
    agent_card: AgentCard,
    transport_protocol: TransportProtocol,
    timeout: int,
    headers: MutableMapping[str, str],
    streaming: bool,
    auth: AuthScheme | None = None,
    use_polling: bool = False,
    push_notification_config: PushNotificationConfig | None = None,
) -> AsyncIterator[Client]:
    """Create and configure an A2A client.

    Args:
        agent_card: The A2A agent card
        transport_protocol: Transport protocol to use
        timeout: Request timeout in seconds
        headers: HTTP headers (already with auth applied)
        streaming: Enable streaming responses
        auth: Optional AuthScheme for client configuration
        use_polling: Enable polling mode
        push_notification_config: Optional push notification config to include in requests

    Yields:
        Configured A2A client instance
    """

    async with httpx.AsyncClient(
        timeout=timeout,
        headers=headers,
    ) as httpx_client:
        if auth and isinstance(auth, (HTTPDigestAuth, APIKeyAuth)):
            configure_auth_client(auth, httpx_client)

        push_configs: list[A2APushNotificationConfig] = []
        if push_notification_config is not None:
            push_configs.append(
                A2APushNotificationConfig(
                    url=str(push_notification_config.url),
                    id=push_notification_config.id,
                    token=push_notification_config.token,
                    authentication=push_notification_config.authentication,
                )
            )

        config = ClientConfig(
            httpx_client=httpx_client,
            supported_transports=[str(transport_protocol.value)],
            streaming=streaming and not use_polling,
            polling=use_polling,
            accepted_output_modes=["application/json"],
            push_notification_configs=push_configs,
        )

        factory = ClientFactory(config)
        client = factory.create(agent_card)
        yield client


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
                description="Set to false when the remote agent has answered your question - extract their answer and return it as your final message. Set to true ONLY if you need to ask a NEW, DIFFERENT question. NEVER repeat the same request - if the conversation history shows the agent already answered, set is_a2a=false immediately."
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
