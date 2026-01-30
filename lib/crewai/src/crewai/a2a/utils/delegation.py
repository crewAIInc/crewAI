"""A2A delegation utilities for executing tasks on remote agents."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncIterator, Callable, MutableMapping
from contextlib import asynccontextmanager
import logging
from typing import TYPE_CHECKING, Any, Final, Literal
import uuid

from a2a.client import Client, ClientConfig, ClientFactory
from a2a.types import (
    AgentCard,
    FilePart,
    FileWithBytes,
    Message,
    Part,
    PushNotificationConfig as A2APushNotificationConfig,
    Role,
    TextPart,
)
import httpx
from pydantic import BaseModel

from crewai.a2a.auth.client_schemes import APIKeyAuth, HTTPDigestAuth
from crewai.a2a.auth.utils import (
    _auth_store,
    configure_auth_client,
    validate_auth_against_agent_card,
)
from crewai.a2a.config import ClientTransportConfig, GRPCClientConfig
from crewai.a2a.extensions.registry import (
    ExtensionsMiddleware,
    validate_required_extensions,
)
from crewai.a2a.task_helpers import TaskStateResult
from crewai.a2a.types import (
    HANDLER_REGISTRY,
    HandlerType,
    PartsDict,
    PartsMetadataDict,
    TransportType,
)
from crewai.a2a.updates import (
    PollingConfig,
    PushNotificationConfig,
    StreamingHandler,
    UpdateConfig,
)
from crewai.a2a.utils.agent_card import (
    _afetch_agent_card_cached,
    _get_tls_verify,
    _prepare_auth_headers,
)
from crewai.a2a.utils.content_type import (
    DEFAULT_CLIENT_OUTPUT_MODES,
    negotiate_content_types,
)
from crewai.a2a.utils.transport import (
    NegotiatedTransport,
    TransportNegotiationError,
    negotiate_transport,
)
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.a2a_events import (
    A2AConversationStartedEvent,
    A2ADelegationCompletedEvent,
    A2ADelegationStartedEvent,
    A2AMessageSentEvent,
)


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from a2a.types import Message

    from crewai.a2a.auth.client_schemes import ClientAuthScheme


_DEFAULT_TRANSPORT: Final[TransportType] = "JSONRPC"


def _create_file_parts(input_files: dict[str, Any] | None) -> list[Part]:
    """Convert FileInput dictionary to FilePart objects.

    Args:
        input_files: Dictionary mapping names to FileInput objects.

    Returns:
        List of Part objects containing FilePart data.
    """
    if not input_files:
        return []

    try:
        import crewai_files  # noqa: F401
    except ImportError:
        logger.debug("crewai_files not installed, skipping file parts")
        return []

    parts: list[Part] = []
    for name, file_input in input_files.items():
        content_bytes = file_input.read()
        content_base64 = base64.b64encode(content_bytes).decode()
        file_with_bytes = FileWithBytes(
            bytes=content_base64,
            mimeType=file_input.content_type,
            name=file_input.filename or name,
        )
        parts.append(Part(root=FilePart(file=file_with_bytes)))

    return parts


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


def execute_a2a_delegation(
    endpoint: str,
    auth: ClientAuthScheme | None,
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
    from_task: Any | None = None,
    from_agent: Any | None = None,
    skill_id: str | None = None,
    client_extensions: list[str] | None = None,
    transport: ClientTransportConfig | None = None,
    accepted_output_modes: list[str] | None = None,
    input_files: dict[str, Any] | None = None,
) -> TaskStateResult:
    """Execute a task delegation to a remote A2A agent synchronously.

    WARNING: This function blocks the entire thread by creating and running a new
    event loop. Prefer using 'await aexecute_a2a_delegation()' in async contexts
    for better performance and resource efficiency.

    This is a synchronous wrapper around aexecute_a2a_delegation that creates a
    new event loop to run the async implementation. It is provided for compatibility
    with synchronous code paths only.

    Args:
        endpoint: A2A agent endpoint URL (AgentCard URL).
        auth: Optional ClientAuthScheme for authentication.
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
        from_task: Optional CrewAI Task object for event metadata.
        from_agent: Optional CrewAI Agent object for event metadata.
        skill_id: Optional skill ID to target a specific agent capability.
        client_extensions: A2A protocol extension URIs the client supports.
        transport: Transport configuration (preferred, supported transports, gRPC settings).
        accepted_output_modes: MIME types the client can accept in responses.
        input_files: Optional dictionary of files to send to remote agent.

    Returns:
        TaskStateResult with status, result/error, history, and agent_card.

    Raises:
        RuntimeError: If called from an async context with a running event loop.
    """
    try:
        asyncio.get_running_loop()
        raise RuntimeError(
            "execute_a2a_delegation() cannot be called from an async context. "
            "Use 'await aexecute_a2a_delegation()' instead."
        )
    except RuntimeError as e:
        if "no running event loop" not in str(e).lower():
            raise

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
                from_task=from_task,
                from_agent=from_agent,
                skill_id=skill_id,
                client_extensions=client_extensions,
                transport=transport,
                accepted_output_modes=accepted_output_modes,
                input_files=input_files,
            )
        )
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            loop.close()


async def aexecute_a2a_delegation(
    endpoint: str,
    auth: ClientAuthScheme | None,
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
    from_task: Any | None = None,
    from_agent: Any | None = None,
    skill_id: str | None = None,
    client_extensions: list[str] | None = None,
    transport: ClientTransportConfig | None = None,
    accepted_output_modes: list[str] | None = None,
    input_files: dict[str, Any] | None = None,
) -> TaskStateResult:
    """Execute a task delegation to a remote A2A agent asynchronously.

    Native async implementation with multi-turn support. Use this when running
    in an async context (e.g., with Crew.akickoff() or agent.aexecute_task()).

    Args:
        endpoint: A2A agent endpoint URL.
        auth: Optional ClientAuthScheme for authentication.
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
        from_task: Optional CrewAI Task object for event metadata.
        from_agent: Optional CrewAI Agent object for event metadata.
        skill_id: Optional skill ID to target a specific agent capability.
        client_extensions: A2A protocol extension URIs the client supports.
        transport: Transport configuration (preferred, supported transports, gRPC settings).
        accepted_output_modes: MIME types the client can accept in responses.
        input_files: Optional dictionary of files to send to remote agent.

    Returns:
        TaskStateResult with status, result/error, history, and agent_card.
    """
    if conversation_history is None:
        conversation_history = []

    is_multiturn = len(conversation_history) > 0
    if turn_number is None:
        turn_number = len([m for m in conversation_history if m.role == Role.user]) + 1

    try:
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
            from_task=from_task,
            from_agent=from_agent,
            skill_id=skill_id,
            client_extensions=client_extensions,
            transport=transport,
            accepted_output_modes=accepted_output_modes,
            input_files=input_files,
        )
    except Exception as e:
        crewai_event_bus.emit(
            agent_branch,
            A2ADelegationCompletedEvent(
                status="failed",
                result=None,
                error=str(e),
                context_id=context_id,
                is_multiturn=is_multiturn,
                endpoint=endpoint,
                metadata=metadata,
                extensions=list(extensions.keys()) if extensions else None,
                from_task=from_task,
                from_agent=from_agent,
            ),
        )
        raise

    agent_card_data = result.get("agent_card")
    crewai_event_bus.emit(
        agent_branch,
        A2ADelegationCompletedEvent(
            status=result["status"],
            result=result.get("result"),
            error=result.get("error"),
            context_id=context_id,
            is_multiturn=is_multiturn,
            endpoint=endpoint,
            a2a_agent_name=result.get("a2a_agent_name"),
            agent_card=agent_card_data,
            provider=agent_card_data.get("provider") if agent_card_data else None,
            metadata=metadata,
            extensions=list(extensions.keys()) if extensions else None,
            from_task=from_task,
            from_agent=from_agent,
        ),
    )

    return result


async def _aexecute_a2a_delegation_impl(
    endpoint: str,
    auth: ClientAuthScheme | None,
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
    from_task: Any | None = None,
    from_agent: Any | None = None,
    skill_id: str | None = None,
    client_extensions: list[str] | None = None,
    transport: ClientTransportConfig | None = None,
    accepted_output_modes: list[str] | None = None,
    input_files: dict[str, Any] | None = None,
) -> TaskStateResult:
    """Internal async implementation of A2A delegation."""
    if transport is None:
        transport = ClientTransportConfig()
    if auth:
        auth_data = auth.model_dump_json(
            exclude={
                "_access_token",
                "_token_expires_at",
                "_refresh_token",
                "_authorization_callback",
            }
        )
        auth_hash = _auth_store.compute_key(type(auth).__name__, auth_data)
    else:
        auth_hash = _auth_store.compute_key("none", endpoint)
    _auth_store.set(auth_hash, auth)
    agent_card = await _afetch_agent_card_cached(
        endpoint=endpoint, auth_hash=auth_hash, timeout=timeout
    )

    validate_auth_against_agent_card(agent_card, auth)

    unsupported_exts = validate_required_extensions(agent_card, client_extensions)
    if unsupported_exts:
        ext_uris = [ext.uri for ext in unsupported_exts]
        raise ValueError(
            f"Agent requires extensions not supported by client: {ext_uris}"
        )

    negotiated: NegotiatedTransport | None = None
    effective_transport: TransportType = transport.preferred or _DEFAULT_TRANSPORT
    effective_url = endpoint

    client_transports: list[str] = (
        list(transport.supported) if transport.supported else [_DEFAULT_TRANSPORT]
    )

    try:
        negotiated = negotiate_transport(
            agent_card=agent_card,
            client_supported_transports=client_transports,
            client_preferred_transport=transport.preferred,
            endpoint=endpoint,
            a2a_agent_name=agent_card.name,
        )
        effective_transport = negotiated.transport  # type: ignore[assignment]
        effective_url = negotiated.url
    except TransportNegotiationError as e:
        logger.warning(
            "Transport negotiation failed, using fallback",
            extra={
                "error": str(e),
                "fallback_transport": effective_transport,
                "fallback_url": effective_url,
                "endpoint": endpoint,
                "client_transports": client_transports,
                "server_transports": [
                    iface.transport for iface in agent_card.additional_interfaces or []
                ]
                + [agent_card.preferred_transport or "JSONRPC"],
            },
        )

    effective_output_modes = accepted_output_modes or DEFAULT_CLIENT_OUTPUT_MODES.copy()

    content_negotiated = negotiate_content_types(
        agent_card=agent_card,
        client_output_modes=accepted_output_modes,
        skill_name=skill_id,
        endpoint=endpoint,
        a2a_agent_name=agent_card.name,
    )
    if content_negotiated.output_modes:
        effective_output_modes = content_negotiated.output_modes

    headers, _ = await _prepare_auth_headers(auth, timeout)

    a2a_agent_name = None
    if agent_card.name:
        a2a_agent_name = agent_card.name

    agent_card_dict = agent_card.model_dump(exclude_none=True)
    crewai_event_bus.emit(
        agent_branch,
        A2ADelegationStartedEvent(
            endpoint=endpoint,
            task_description=task_description,
            agent_id=agent_id or endpoint,
            context_id=context_id,
            is_multiturn=is_multiturn,
            turn_number=turn_number,
            a2a_agent_name=a2a_agent_name,
            agent_card=agent_card_dict,
            protocol_version=agent_card.protocol_version,
            provider=agent_card_dict.get("provider"),
            skill_id=skill_id,
            metadata=metadata,
            extensions=list(extensions.keys()) if extensions else None,
            from_task=from_task,
            from_agent=from_agent,
        ),
    )

    if turn_number == 1:
        agent_id_for_event = agent_id or endpoint
        crewai_event_bus.emit(
            agent_branch,
            A2AConversationStartedEvent(
                agent_id=agent_id_for_event,
                endpoint=endpoint,
                context_id=context_id,
                a2a_agent_name=a2a_agent_name,
                agent_card=agent_card_dict,
                protocol_version=agent_card.protocol_version,
                provider=agent_card_dict.get("provider"),
                skill_id=skill_id,
                reference_task_ids=reference_task_ids,
                metadata=metadata,
                extensions=list(extensions.keys()) if extensions else None,
                from_task=from_task,
                from_agent=from_agent,
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

    message_metadata = metadata.copy() if metadata else {}
    if skill_id:
        message_metadata["skill_id"] = skill_id

    parts_list: list[Part] = [Part(root=TextPart(**parts))]
    parts_list.extend(_create_file_parts(input_files))

    message = Message(
        role=Role.user,
        message_id=str(uuid.uuid4()),
        parts=parts_list,
        context_id=context_id,
        task_id=task_id,
        reference_task_ids=reference_task_ids,
        metadata=message_metadata if message_metadata else None,
        extensions=extensions,
    )

    new_messages: list[Message] = [*conversation_history, message]
    crewai_event_bus.emit(
        None,
        A2AMessageSentEvent(
            message=message_text,
            turn_number=turn_number,
            context_id=context_id,
            message_id=message.message_id,
            is_multiturn=is_multiturn,
            agent_role=agent_role,
            endpoint=endpoint,
            a2a_agent_name=a2a_agent_name,
            skill_id=skill_id,
            metadata=message_metadata if message_metadata else None,
            extensions=list(extensions.keys()) if extensions else None,
            from_task=from_task,
            from_agent=from_agent,
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
        "a2a_agent_name": a2a_agent_name,
        "from_task": from_task,
        "from_agent": from_agent,
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

    client_agent_card = agent_card
    if effective_url != agent_card.url:
        client_agent_card = agent_card.model_copy(update={"url": effective_url})

    async with _create_a2a_client(
        agent_card=client_agent_card,
        transport_protocol=effective_transport,
        timeout=timeout,
        headers=headers,
        streaming=use_streaming,
        auth=auth,
        use_polling=use_polling,
        push_notification_config=push_config_for_client,
        client_extensions=client_extensions,
        accepted_output_modes=effective_output_modes,  # type: ignore[arg-type]
        grpc_config=transport.grpc,
    ) as client:
        result = await handler.execute(
            client=client,
            message=message,
            new_messages=new_messages,
            agent_card=agent_card,
            **handler_kwargs,
        )
        result["a2a_agent_name"] = a2a_agent_name
        result["agent_card"] = agent_card.model_dump(exclude_none=True)
        return result


def _normalize_grpc_metadata(
    metadata: tuple[tuple[str, str], ...] | None,
) -> tuple[tuple[str, str], ...] | None:
    """Lowercase all gRPC metadata keys.

    gRPC requires lowercase metadata keys, but some libraries (like the A2A SDK)
    use mixed-case headers like 'X-A2A-Extensions'. This normalizes them.
    """
    if metadata is None:
        return None
    return tuple((key.lower(), value) for key, value in metadata)


def _create_grpc_interceptors(
    auth_metadata: list[tuple[str, str]] | None = None,
) -> list[Any]:
    """Create gRPC interceptors for metadata normalization and auth injection.

    Args:
        auth_metadata: Optional auth metadata to inject into all calls.
            Used for insecure channels that need auth (non-localhost without TLS).

    Returns a list of interceptors that lowercase metadata keys for gRPC
    compatibility. Must be called after grpc is imported.
    """
    import grpc.aio  # type: ignore[import-untyped]

    def _merge_metadata(
        existing: tuple[tuple[str, str], ...] | None,
        auth: list[tuple[str, str]] | None,
    ) -> tuple[tuple[str, str], ...] | None:
        """Merge existing metadata with auth metadata and normalize keys."""
        merged: list[tuple[str, str]] = []
        if existing:
            merged.extend(existing)
        if auth:
            merged.extend(auth)
        if not merged:
            return None
        return tuple((key.lower(), value) for key, value in merged)

    def _inject_metadata(client_call_details: Any) -> Any:
        """Inject merged metadata into call details."""
        return client_call_details._replace(
            metadata=_merge_metadata(client_call_details.metadata, auth_metadata)
        )

    class MetadataUnaryUnary(grpc.aio.UnaryUnaryClientInterceptor):  # type: ignore[misc,no-any-unimported]
        """Interceptor for unary-unary calls that injects auth metadata."""

        async def intercept_unary_unary(  # type: ignore[no-untyped-def]
            self, continuation, client_call_details, request
        ):
            """Intercept unary-unary call and inject metadata."""
            return await continuation(_inject_metadata(client_call_details), request)

    class MetadataUnaryStream(grpc.aio.UnaryStreamClientInterceptor):  # type: ignore[misc,no-any-unimported]
        """Interceptor for unary-stream calls that injects auth metadata."""

        async def intercept_unary_stream(  # type: ignore[no-untyped-def]
            self, continuation, client_call_details, request
        ):
            """Intercept unary-stream call and inject metadata."""
            return await continuation(_inject_metadata(client_call_details), request)

    class MetadataStreamUnary(grpc.aio.StreamUnaryClientInterceptor):  # type: ignore[misc,no-any-unimported]
        """Interceptor for stream-unary calls that injects auth metadata."""

        async def intercept_stream_unary(  # type: ignore[no-untyped-def]
            self, continuation, client_call_details, request_iterator
        ):
            """Intercept stream-unary call and inject metadata."""
            return await continuation(
                _inject_metadata(client_call_details), request_iterator
            )

    class MetadataStreamStream(grpc.aio.StreamStreamClientInterceptor):  # type: ignore[misc,no-any-unimported]
        """Interceptor for stream-stream calls that injects auth metadata."""

        async def intercept_stream_stream(  # type: ignore[no-untyped-def]
            self, continuation, client_call_details, request_iterator
        ):
            """Intercept stream-stream call and inject metadata."""
            return await continuation(
                _inject_metadata(client_call_details), request_iterator
            )

    return [
        MetadataUnaryUnary(),
        MetadataUnaryStream(),
        MetadataStreamUnary(),
        MetadataStreamStream(),
    ]


def _create_grpc_channel_factory(
    grpc_config: GRPCClientConfig,
    auth: ClientAuthScheme | None = None,
) -> Callable[[str], Any]:
    """Create a gRPC channel factory with the given configuration.

    Args:
        grpc_config: gRPC client configuration with channel options.
        auth: Optional ClientAuthScheme for TLS and auth configuration.

    Returns:
        A callable that creates gRPC channels from URLs.
    """
    try:
        import grpc
    except ImportError as e:
        raise ImportError(
            "gRPC transport requires grpcio. Install with: pip install a2a-sdk[grpc]"
        ) from e

    auth_metadata: list[tuple[str, str]] = []

    if auth is not None:
        from crewai.a2a.auth.client_schemes import (
            APIKeyAuth,
            BearerTokenAuth,
            HTTPBasicAuth,
            HTTPDigestAuth,
            OAuth2AuthorizationCode,
            OAuth2ClientCredentials,
        )

        if isinstance(auth, HTTPDigestAuth):
            raise ValueError(
                "HTTPDigestAuth is not supported with gRPC transport. "
                "Digest authentication requires HTTP challenge-response flow. "
                "Use BearerTokenAuth, HTTPBasicAuth, APIKeyAuth (header), or OAuth2 instead."
            )
        if isinstance(auth, APIKeyAuth) and auth.location in ("query", "cookie"):
            raise ValueError(
                f"APIKeyAuth with location='{auth.location}' is not supported with gRPC transport. "
                "gRPC only supports header-based authentication. "
                "Use APIKeyAuth with location='header' instead."
            )

        if isinstance(auth, BearerTokenAuth):
            auth_metadata.append(("authorization", f"Bearer {auth.token}"))
        elif isinstance(auth, HTTPBasicAuth):
            import base64

            basic_credentials = f"{auth.username}:{auth.password}"
            encoded = base64.b64encode(basic_credentials.encode()).decode()
            auth_metadata.append(("authorization", f"Basic {encoded}"))
        elif isinstance(auth, APIKeyAuth) and auth.location == "header":
            header_name = auth.name.lower()
            auth_metadata.append((header_name, auth.api_key))
        elif isinstance(auth, (OAuth2ClientCredentials, OAuth2AuthorizationCode)):
            if auth._access_token:
                auth_metadata.append(("authorization", f"Bearer {auth._access_token}"))

    def factory(url: str) -> Any:
        """Create a gRPC channel for the given URL."""
        target = url
        use_tls = False

        if url.startswith("grpcs://"):
            target = url[8:]
            use_tls = True
        elif url.startswith("grpc://"):
            target = url[7:]
        elif url.startswith("https://"):
            target = url[8:]
            use_tls = True
        elif url.startswith("http://"):
            target = url[7:]

        options: list[tuple[str, Any]] = []
        if grpc_config.max_send_message_length is not None:
            options.append(
                ("grpc.max_send_message_length", grpc_config.max_send_message_length)
            )
        if grpc_config.max_receive_message_length is not None:
            options.append(
                (
                    "grpc.max_receive_message_length",
                    grpc_config.max_receive_message_length,
                )
            )
        if grpc_config.keepalive_time_ms is not None:
            options.append(("grpc.keepalive_time_ms", grpc_config.keepalive_time_ms))
        if grpc_config.keepalive_timeout_ms is not None:
            options.append(
                ("grpc.keepalive_timeout_ms", grpc_config.keepalive_timeout_ms)
            )

        channel_credentials = None
        if auth and hasattr(auth, "tls") and auth.tls:
            channel_credentials = auth.tls.get_grpc_credentials()
        elif use_tls:
            channel_credentials = grpc.ssl_channel_credentials()

        if channel_credentials and auth_metadata:

            class AuthMetadataPlugin(grpc.AuthMetadataPlugin):  # type: ignore[misc,no-any-unimported]
                """gRPC auth metadata plugin that adds auth headers as metadata."""

                def __init__(self, metadata: list[tuple[str, str]]) -> None:
                    self._metadata = tuple(metadata)

                def __call__(  # type: ignore[no-any-unimported]
                    self,
                    context: grpc.AuthMetadataContext,
                    callback: grpc.AuthMetadataPluginCallback,
                ) -> None:
                    callback(self._metadata, None)

            call_creds = grpc.metadata_call_credentials(
                AuthMetadataPlugin(auth_metadata)
            )
            credentials = grpc.composite_channel_credentials(
                channel_credentials, call_creds
            )
            interceptors = _create_grpc_interceptors()
            return grpc.aio.secure_channel(
                target, credentials, options=options or None, interceptors=interceptors
            )
        if channel_credentials:
            interceptors = _create_grpc_interceptors()
            return grpc.aio.secure_channel(
                target,
                channel_credentials,
                options=options or None,
                interceptors=interceptors,
            )
        interceptors = _create_grpc_interceptors(
            auth_metadata=auth_metadata if auth_metadata else None
        )
        return grpc.aio.insecure_channel(
            target, options=options or None, interceptors=interceptors
        )

    return factory


@asynccontextmanager
async def _create_a2a_client(
    agent_card: AgentCard,
    transport_protocol: Literal["JSONRPC", "GRPC", "HTTP+JSON"],
    timeout: int,
    headers: MutableMapping[str, str],
    streaming: bool,
    auth: ClientAuthScheme | None = None,
    use_polling: bool = False,
    push_notification_config: PushNotificationConfig | None = None,
    client_extensions: list[str] | None = None,
    accepted_output_modes: list[str] | None = None,
    grpc_config: GRPCClientConfig | None = None,
) -> AsyncIterator[Client]:
    """Create and configure an A2A client.

    Args:
        agent_card: The A2A agent card.
        transport_protocol: Transport protocol to use.
        timeout: Request timeout in seconds.
        headers: HTTP headers (already with auth applied).
        streaming: Enable streaming responses.
        auth: Optional ClientAuthScheme for client configuration.
        use_polling: Enable polling mode.
        push_notification_config: Optional push notification config.
        client_extensions: A2A protocol extension URIs to declare support for.
        accepted_output_modes: MIME types the client can accept in responses.
        grpc_config: Optional gRPC client configuration.

    Yields:
        Configured A2A client instance.
    """
    verify = _get_tls_verify(auth)
    async with httpx.AsyncClient(
        timeout=timeout,
        headers=headers,
        verify=verify,
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

        grpc_channel_factory = None
        if transport_protocol == "GRPC":
            grpc_channel_factory = _create_grpc_channel_factory(
                grpc_config or GRPCClientConfig(),
                auth=auth,
            )

        config = ClientConfig(
            httpx_client=httpx_client,
            supported_transports=[transport_protocol],
            streaming=streaming and not use_polling,
            polling=use_polling,
            accepted_output_modes=accepted_output_modes or DEFAULT_CLIENT_OUTPUT_MODES,  # type: ignore[arg-type]
            push_notification_configs=push_configs,
            grpc_channel_factory=grpc_channel_factory,
        )

        factory = ClientFactory(config)
        client = factory.create(agent_card)

        if client_extensions:
            await client.add_request_middleware(ExtensionsMiddleware(client_extensions))

        yield client
