"""Response model utilities for A2A agent interactions."""

from __future__ import annotations

from typing import TypeAlias

from pydantic import BaseModel, Field, create_model

from crewai.a2a.config import A2AClientConfig, A2AConfig, A2AServerConfig
from crewai.types.utils import create_literals_from_strings


A2AConfigTypes: TypeAlias = A2AConfig | A2AServerConfig | A2AClientConfig
A2AClientConfigTypes: TypeAlias = A2AConfig | A2AClientConfig


def create_agent_response_model(agent_ids: tuple[str, ...]) -> type[BaseModel] | None:
    """Create a dynamic AgentResponse model with Literal types for agent IDs.

    Args:
        agent_ids: List of available A2A agent IDs.

    Returns:
        Dynamically created Pydantic model with Literal-constrained a2a_ids field,
        or None if agent_ids is empty.
    """
    if not agent_ids:
        return None

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
    a2a_config: list[A2AConfigTypes] | A2AConfigTypes | None,
) -> tuple[list[A2AClientConfigTypes], tuple[str, ...]]:
    """Extract A2A agent IDs from A2A configuration.

    Filters out A2AServerConfig since it doesn't have an endpoint for delegation.

    Args:
        a2a_config: A2A configuration (any type).

    Returns:
        Tuple of client A2A configs list and agent endpoint IDs.
    """
    if a2a_config is None:
        return [], ()

    configs: list[A2AConfigTypes]
    if isinstance(a2a_config, (A2AConfig, A2AClientConfig, A2AServerConfig)):
        configs = [a2a_config]
    else:
        configs = a2a_config

    # Filter to only client configs (those with endpoint)
    client_configs: list[A2AClientConfigTypes] = [
        config for config in configs if isinstance(config, (A2AConfig, A2AClientConfig))
    ]

    return client_configs, tuple(config.endpoint for config in client_configs)


def get_a2a_agents_and_response_model(
    a2a_config: list[A2AConfigTypes] | A2AConfigTypes | None,
) -> tuple[list[A2AClientConfigTypes], type[BaseModel] | None]:
    """Get A2A agent configs and response model.

    Args:
        a2a_config: A2A configuration (any type).

    Returns:
        Tuple of client A2A configs and response model.
    """
    a2a_agents, agent_ids = extract_a2a_agent_ids_from_config(a2a_config=a2a_config)

    return a2a_agents, create_agent_response_model(agent_ids)
