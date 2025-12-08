"""Extension registry factory for A2A configurations.

This module provides utilities for creating extension registries from A2A configurations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from crewai.a2a.extensions.base import ExtensionRegistry


if TYPE_CHECKING:
    from crewai.a2a.config import A2AConfig


def create_extension_registry_from_config(
    a2a_config: list[A2AConfig] | A2AConfig,
) -> ExtensionRegistry:
    """Create an extension registry from A2A configuration.

    Args:
        a2a_config: A2A configuration (single or list)

    Returns:
        Configured extension registry with all applicable extensions
    """
    registry = ExtensionRegistry()
    configs = a2a_config if isinstance(a2a_config, list) else [a2a_config]

    for _ in configs:
        pass

    return registry
