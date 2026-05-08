"""Helpers for extracting A2A client configurations."""

from __future__ import annotations

from typing import TypeAlias

from crewai.a2a.config import A2AClientConfig, A2AConfig, A2AServerConfig


A2AConfigTypes: TypeAlias = A2AConfig | A2AServerConfig | A2AClientConfig
A2AClientConfigTypes: TypeAlias = A2AConfig | A2AClientConfig


def extract_a2a_client_configs(
    a2a_config: list[A2AConfigTypes] | A2AConfigTypes | None,
) -> list[A2AClientConfigTypes]:
    """Return the client-side A2A configs from a possibly-mixed config list.

    Filters out :class:`A2AServerConfig`, which has no endpoint to delegate to.
    """
    if a2a_config is None:
        return []

    configs: list[A2AConfigTypes]
    if isinstance(a2a_config, (A2AConfig, A2AClientConfig, A2AServerConfig)):
        configs = [a2a_config]
    else:
        configs = a2a_config

    return [
        config for config in configs if isinstance(config, (A2AConfig, A2AClientConfig))
    ]
