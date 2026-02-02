"""AgentCard utilities for A2A client and server operations."""

from __future__ import annotations

import asyncio
from collections.abc import MutableMapping
from functools import lru_cache
import ssl
import time
from types import MethodType
from typing import TYPE_CHECKING

from a2a.client.errors import A2AClientHTTPError
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from aiocache import cached  # type: ignore[import-untyped]
from aiocache.serializers import PickleSerializer  # type: ignore[import-untyped]
import httpx

from crewai.a2a.auth.client_schemes import APIKeyAuth, HTTPDigestAuth
from crewai.a2a.auth.utils import (
    _auth_store,
    configure_auth_client,
    retry_on_401,
)
from crewai.a2a.config import A2AServerConfig
from crewai.crew import Crew
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.a2a_events import (
    A2AAgentCardFetchedEvent,
    A2AAuthenticationFailedEvent,
    A2AConnectionErrorEvent,
)


if TYPE_CHECKING:
    from crewai.a2a.auth.client_schemes import ClientAuthScheme
    from crewai.agent import Agent
    from crewai.task import Task


def _get_tls_verify(auth: ClientAuthScheme | None) -> ssl.SSLContext | bool | str:
    """Get TLS verify parameter from auth scheme.

    Args:
        auth: Optional authentication scheme with TLS config.

    Returns:
        SSL context, CA cert path, True for default verification,
        or False if verification disabled.
    """
    if auth and auth.tls:
        return auth.tls.get_httpx_ssl_context()
    return True


async def _prepare_auth_headers(
    auth: ClientAuthScheme | None,
    timeout: int,
) -> tuple[MutableMapping[str, str], ssl.SSLContext | bool | str]:
    """Prepare authentication headers and TLS verification settings.

    Args:
        auth: Optional authentication scheme.
        timeout: Request timeout in seconds.

    Returns:
        Tuple of (headers dict, TLS verify setting).
    """
    headers: MutableMapping[str, str] = {}
    verify = _get_tls_verify(auth)
    if auth:
        async with httpx.AsyncClient(
            timeout=timeout, verify=verify
        ) as temp_auth_client:
            if isinstance(auth, (HTTPDigestAuth, APIKeyAuth)):
                configure_auth_client(auth, temp_auth_client)
            headers = await auth.apply_auth(temp_auth_client, {})
    return headers, verify


def _get_server_config(agent: Agent) -> A2AServerConfig | None:
    """Get A2AServerConfig from an agent's a2a configuration.

    Args:
        agent: The Agent instance to check.

    Returns:
        A2AServerConfig if present, None otherwise.
    """
    if agent.a2a is None:
        return None
    if isinstance(agent.a2a, A2AServerConfig):
        return agent.a2a
    if isinstance(agent.a2a, list):
        for config in agent.a2a:
            if isinstance(config, A2AServerConfig):
                return config
    return None


def fetch_agent_card(
    endpoint: str,
    auth: ClientAuthScheme | None = None,
    timeout: int = 30,
    use_cache: bool = True,
    cache_ttl: int = 300,
) -> AgentCard:
    """Fetch AgentCard from an A2A endpoint with optional caching.

    Args:
        endpoint: A2A agent endpoint URL (AgentCard URL).
        auth: Optional ClientAuthScheme for authentication.
        timeout: Request timeout in seconds.
        use_cache: Whether to use caching (default True).
        cache_ttl: Cache TTL in seconds (default 300 = 5 minutes).

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
            auth_hash = _auth_store.compute_key(type(auth).__name__, auth_data)
        else:
            auth_hash = _auth_store.compute_key("none", "")
        _auth_store.set(auth_hash, auth)
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
    auth: ClientAuthScheme | None = None,
    timeout: int = 30,
    use_cache: bool = True,
) -> AgentCard:
    """Fetch AgentCard from an A2A endpoint asynchronously.

    Native async implementation. Use this when running in an async context.

    Args:
        endpoint: A2A agent endpoint URL (AgentCard URL).
        auth: Optional ClientAuthScheme for authentication.
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
            auth_hash = _auth_store.compute_key(type(auth).__name__, auth_data)
        else:
            auth_hash = _auth_store.compute_key("none", "")
        _auth_store.set(auth_hash, auth)
        agent_card: AgentCard = await _afetch_agent_card_cached(
            endpoint, auth_hash, timeout
        )
        return agent_card

    return await _afetch_agent_card_impl(endpoint=endpoint, auth=auth, timeout=timeout)


@lru_cache()
def _fetch_agent_card_cached(
    endpoint: str,
    auth_hash: str,
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


@cached(ttl=300, serializer=PickleSerializer())  # type: ignore[untyped-decorator]
async def _afetch_agent_card_cached(
    endpoint: str,
    auth_hash: str,
    timeout: int,
) -> AgentCard:
    """Cached async implementation of AgentCard fetching."""
    auth = _auth_store.get(auth_hash)
    return await _afetch_agent_card_impl(endpoint=endpoint, auth=auth, timeout=timeout)


async def _afetch_agent_card_impl(
    endpoint: str,
    auth: ClientAuthScheme | None,
    timeout: int,
) -> AgentCard:
    """Internal async implementation of AgentCard fetching."""
    start_time = time.perf_counter()

    if "/.well-known/agent-card.json" in endpoint:
        base_url = endpoint.replace("/.well-known/agent-card.json", "")
        agent_card_path = "/.well-known/agent-card.json"
    else:
        url_parts = endpoint.split("/", 3)
        base_url = f"{url_parts[0]}//{url_parts[2]}"
        agent_card_path = (
            f"/{url_parts[3]}"
            if len(url_parts) > 3 and url_parts[3]
            else "/.well-known/agent-card.json"
        )

    headers, verify = await _prepare_auth_headers(auth, timeout)

    async with httpx.AsyncClient(
        timeout=timeout, headers=headers, verify=verify
    ) as temp_client:
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

            agent_card = AgentCard.model_validate(response.json())
            fetch_time_ms = (time.perf_counter() - start_time) * 1000
            agent_card_dict = agent_card.model_dump(exclude_none=True)

            crewai_event_bus.emit(
                None,
                A2AAgentCardFetchedEvent(
                    endpoint=endpoint,
                    a2a_agent_name=agent_card.name,
                    agent_card=agent_card_dict,
                    protocol_version=agent_card.protocol_version,
                    provider=agent_card_dict.get("provider"),
                    cached=False,
                    fetch_time_ms=fetch_time_ms,
                ),
            )

            return agent_card

        except httpx.HTTPStatusError as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            response_body = e.response.text[:1000] if e.response.text else None

            if e.response.status_code == 401:
                error_details = ["Authentication failed"]
                www_auth = e.response.headers.get("WWW-Authenticate")
                if www_auth:
                    error_details.append(f"WWW-Authenticate: {www_auth}")
                if not auth:
                    error_details.append("No auth scheme provided")
                msg = " | ".join(error_details)

                auth_type = type(auth).__name__ if auth else None
                crewai_event_bus.emit(
                    None,
                    A2AAuthenticationFailedEvent(
                        endpoint=endpoint,
                        auth_type=auth_type,
                        error=msg,
                        status_code=401,
                        metadata={
                            "elapsed_ms": elapsed_ms,
                            "response_body": response_body,
                            "www_authenticate": www_auth,
                            "request_url": str(e.request.url),
                        },
                    ),
                )

                raise A2AClientHTTPError(401, msg) from e

            crewai_event_bus.emit(
                None,
                A2AConnectionErrorEvent(
                    endpoint=endpoint,
                    error=str(e),
                    error_type="http_error",
                    status_code=e.response.status_code,
                    operation="fetch_agent_card",
                    metadata={
                        "elapsed_ms": elapsed_ms,
                        "response_body": response_body,
                        "request_url": str(e.request.url),
                    },
                ),
            )
            raise

        except httpx.TimeoutException as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            crewai_event_bus.emit(
                None,
                A2AConnectionErrorEvent(
                    endpoint=endpoint,
                    error=str(e),
                    error_type="timeout",
                    operation="fetch_agent_card",
                    metadata={
                        "elapsed_ms": elapsed_ms,
                        "timeout_config": timeout,
                        "request_url": str(e.request.url) if e.request else None,
                    },
                ),
            )
            raise

        except httpx.ConnectError as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            crewai_event_bus.emit(
                None,
                A2AConnectionErrorEvent(
                    endpoint=endpoint,
                    error=str(e),
                    error_type="connection_error",
                    operation="fetch_agent_card",
                    metadata={
                        "elapsed_ms": elapsed_ms,
                        "request_url": str(e.request.url) if e.request else None,
                    },
                ),
            )
            raise

        except httpx.RequestError as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            crewai_event_bus.emit(
                None,
                A2AConnectionErrorEvent(
                    endpoint=endpoint,
                    error=str(e),
                    error_type="request_error",
                    operation="fetch_agent_card",
                    metadata={
                        "elapsed_ms": elapsed_ms,
                        "request_url": str(e.request.url) if e.request else None,
                    },
                ),
            )
            raise


def _task_to_skill(task: Task) -> AgentSkill:
    """Convert a CrewAI Task to an A2A AgentSkill.

    Args:
        task: The CrewAI Task to convert.

    Returns:
        AgentSkill representing the task's capability.
    """
    task_name = task.name or task.description[:50]
    task_id = task_name.lower().replace(" ", "_")

    tags: list[str] = []
    if task.agent:
        tags.append(task.agent.role.lower().replace(" ", "-"))

    return AgentSkill(
        id=task_id,
        name=task_name,
        description=task.description,
        tags=tags,
        examples=[task.expected_output] if task.expected_output else None,
    )


def _tool_to_skill(tool_name: str, tool_description: str) -> AgentSkill:
    """Convert an Agent's tool to an A2A AgentSkill.

    Args:
        tool_name: Name of the tool.
        tool_description: Description of what the tool does.

    Returns:
        AgentSkill representing the tool's capability.
    """
    tool_id = tool_name.lower().replace(" ", "_")

    return AgentSkill(
        id=tool_id,
        name=tool_name,
        description=tool_description,
        tags=[tool_name.lower().replace(" ", "-")],
    )


def _crew_to_agent_card(crew: Crew, url: str) -> AgentCard:
    """Generate an A2A AgentCard from a Crew instance.

    Args:
        crew: The Crew instance to generate a card for.
        url: The base URL where this crew will be exposed.

    Returns:
        AgentCard describing the crew's capabilities.
    """
    crew_name = getattr(crew, "name", None) or crew.__class__.__name__

    description_parts: list[str] = []
    crew_description = getattr(crew, "description", None)
    if crew_description:
        description_parts.append(crew_description)
    else:
        agent_roles = [agent.role for agent in crew.agents]
        description_parts.append(
            f"A crew of {len(crew.agents)} agents: {', '.join(agent_roles)}"
        )

    skills = [_task_to_skill(task) for task in crew.tasks]

    return AgentCard(
        name=crew_name,
        description=" ".join(description_parts),
        url=url,
        version="1.0.0",
        capabilities=AgentCapabilities(
            streaming=True,
            push_notifications=True,
        ),
        default_input_modes=["text/plain", "application/json"],
        default_output_modes=["text/plain", "application/json"],
        skills=skills,
    )


def _agent_to_agent_card(agent: Agent, url: str) -> AgentCard:
    """Generate an A2A AgentCard from an Agent instance.

    Uses A2AServerConfig values when available, falling back to agent properties.
    If signing_config is provided, the card will be signed with JWS.

    Args:
        agent: The Agent instance to generate a card for.
        url: The base URL where this agent will be exposed.

    Returns:
        AgentCard describing the agent's capabilities.
    """
    from crewai.a2a.utils.agent_card_signing import sign_agent_card

    server_config = _get_server_config(agent) or A2AServerConfig()

    name = server_config.name or agent.role

    description_parts = [agent.goal]
    if agent.backstory:
        description_parts.append(agent.backstory)
    description = server_config.description or " ".join(description_parts)

    skills: list[AgentSkill] = (
        server_config.skills.copy() if server_config.skills else []
    )

    if not skills:
        if agent.tools:
            for tool in agent.tools:
                tool_name = getattr(tool, "name", None) or tool.__class__.__name__
                tool_desc = getattr(tool, "description", None) or f"Tool: {tool_name}"
                skills.append(_tool_to_skill(tool_name, tool_desc))

        if not skills:
            skills.append(
                AgentSkill(
                    id=agent.role.lower().replace(" ", "_"),
                    name=agent.role,
                    description=agent.goal,
                    tags=[agent.role.lower().replace(" ", "-")],
                )
            )

    capabilities = server_config.capabilities
    if server_config.server_extensions:
        from crewai.a2a.extensions.server import ServerExtensionRegistry

        registry = ServerExtensionRegistry(server_config.server_extensions)
        ext_list = registry.get_agent_extensions()

        existing_exts = list(capabilities.extensions) if capabilities.extensions else []
        existing_uris = {e.uri for e in existing_exts}
        for ext in ext_list:
            if ext.uri not in existing_uris:
                existing_exts.append(ext)

        capabilities = capabilities.model_copy(update={"extensions": existing_exts})

    card = AgentCard(
        name=name,
        description=description,
        url=server_config.url or url,
        version=server_config.version,
        capabilities=capabilities,
        default_input_modes=server_config.default_input_modes,
        default_output_modes=server_config.default_output_modes,
        skills=skills,
        preferred_transport=server_config.transport.preferred,
        protocol_version=server_config.protocol_version,
        provider=server_config.provider,
        documentation_url=server_config.documentation_url,
        icon_url=server_config.icon_url,
        additional_interfaces=server_config.additional_interfaces,
        security=server_config.security,
        security_schemes=server_config.security_schemes,
        supports_authenticated_extended_card=server_config.supports_authenticated_extended_card,
    )

    if server_config.signing_config:
        signature = sign_agent_card(
            card,
            private_key=server_config.signing_config.get_private_key(),
            key_id=server_config.signing_config.key_id,
            algorithm=server_config.signing_config.algorithm,
        )
        card = card.model_copy(update={"signatures": [signature]})
    elif server_config.signatures:
        card = card.model_copy(update={"signatures": server_config.signatures})

    return card


def inject_a2a_server_methods(agent: Agent) -> None:
    """Inject A2A server methods onto an Agent instance.

    Adds a `to_agent_card(url: str) -> AgentCard` method to the agent
    that generates an A2A-compliant AgentCard.

    Only injects if the agent has an A2AServerConfig.

    Args:
        agent: The Agent instance to inject methods onto.
    """
    if _get_server_config(agent) is None:
        return

    def _to_agent_card(self: Agent, url: str) -> AgentCard:
        return _agent_to_agent_card(self, url)

    object.__setattr__(agent, "to_agent_card", MethodType(_to_agent_card, agent))
