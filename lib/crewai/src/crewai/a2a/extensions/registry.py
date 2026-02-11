"""A2A Protocol extension utilities.

This module provides utilities for working with A2A protocol extensions as
defined in the A2A specification. Extensions are capability declarations in
AgentCard.capabilities.extensions using AgentExtension objects, activated
via the X-A2A-Extensions HTTP header.

See: https://a2a-protocol.org/latest/topics/extensions/
"""

from __future__ import annotations

from typing import Any

from a2a.client.middleware import ClientCallContext, ClientCallInterceptor
from a2a.extensions.common import (
    HTTP_EXTENSION_HEADER,
)
from a2a.types import AgentCard, AgentExtension

from crewai.a2a.config import A2AClientConfig, A2AConfig
from crewai.a2a.extensions.base import ExtensionRegistry


def get_extensions_from_config(
    a2a_config: list[A2AConfig | A2AClientConfig] | A2AConfig | A2AClientConfig,
) -> list[str]:
    """Extract extension URIs from A2A configuration.

    Args:
        a2a_config: A2A configuration (single or list).

    Returns:
        Deduplicated list of extension URIs from all configs.
    """
    configs = a2a_config if isinstance(a2a_config, list) else [a2a_config]
    seen: set[str] = set()
    result: list[str] = []

    for config in configs:
        if not isinstance(config, A2AClientConfig):
            continue
        for uri in config.extensions:
            if uri not in seen:
                seen.add(uri)
                result.append(uri)

    return result


class ExtensionsMiddleware(ClientCallInterceptor):
    """Middleware to add X-A2A-Extensions header to requests.

    This middleware adds the extensions header to all outgoing requests,
    declaring which A2A protocol extensions the client supports.
    """

    def __init__(self, extensions: list[str]) -> None:
        """Initialize with extension URIs.

        Args:
            extensions: List of extension URIs the client supports.
        """
        self._extensions = extensions

    async def intercept(
        self,
        method_name: str,
        request_payload: dict[str, Any],
        http_kwargs: dict[str, Any],
        agent_card: AgentCard | None,
        context: ClientCallContext | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Add extensions header to the request.

        Args:
            method_name: The A2A method being called.
            request_payload: The JSON-RPC request payload.
            http_kwargs: HTTP request kwargs (headers, etc).
            agent_card: The target agent's card.
            context: Optional call context.

        Returns:
            Tuple of (request_payload, modified_http_kwargs).
        """
        if self._extensions:
            headers = http_kwargs.setdefault("headers", {})
            headers[HTTP_EXTENSION_HEADER] = ",".join(self._extensions)
        return request_payload, http_kwargs


def validate_required_extensions(
    agent_card: AgentCard,
    client_extensions: list[str] | None,
) -> list[AgentExtension]:
    """Validate that client supports all required extensions from agent.

    Args:
        agent_card: The agent's card with declared extensions.
        client_extensions: Extension URIs the client supports.

    Returns:
        List of unsupported required extensions.

    Raises:
        None - returns list of unsupported extensions for caller to handle.
    """
    unsupported: list[AgentExtension] = []
    client_set = set(client_extensions or [])

    if not agent_card.capabilities or not agent_card.capabilities.extensions:
        return unsupported

    unsupported.extend(
        ext
        for ext in agent_card.capabilities.extensions
        if ext.required and ext.uri not in client_set
    )

    return unsupported


def create_extension_registry_from_config(
    a2a_config: list[A2AConfig | A2AClientConfig] | A2AConfig | A2AClientConfig,
) -> ExtensionRegistry:
    """Create an extension registry from A2A client configuration.

    Extracts client_extensions from each A2AClientConfig and registers them
    with the ExtensionRegistry. These extensions provide CrewAI-specific
    processing hooks (tool injection, prompt augmentation, response processing).

    Note: A2A protocol extensions (URI strings sent via X-A2A-Extensions header)
    are handled separately via get_extensions_from_config() and ExtensionsMiddleware.

    Args:
        a2a_config: A2A configuration (single or list).

    Returns:
        Extension registry with all client_extensions registered.

    Example:
        class LoggingExtension:
            def inject_tools(self, agent): pass
            def extract_state_from_history(self, history): return None
            def augment_prompt(self, prompt, state): return prompt
            def process_response(self, response, state):
                print(f"Response: {response}")
                return response

        config = A2AClientConfig(
            endpoint="https://agent.example.com",
            client_extensions=[LoggingExtension()],
        )
        registry = create_extension_registry_from_config(config)
    """
    registry = ExtensionRegistry()
    configs = a2a_config if isinstance(a2a_config, list) else [a2a_config]

    seen: set[int] = set()

    for config in configs:
        if isinstance(config, (A2AConfig, A2AClientConfig)):
            client_exts = getattr(config, "client_extensions", [])
            for extension in client_exts:
                ext_id = id(extension)
                if ext_id not in seen:
                    seen.add(ext_id)
                    registry.register(extension)

    return registry
