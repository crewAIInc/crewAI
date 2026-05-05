"""Re-exports of shared settings from ``crewai_core.settings``.

Kept as a stable import path for the CLI; new code should import from
``crewai_core.settings`` directly.
"""

from __future__ import annotations

from crewai_core.settings import (
    CLI_SETTINGS_KEYS as CLI_SETTINGS_KEYS,
    DEFAULT_CLI_SETTINGS as DEFAULT_CLI_SETTINGS,
    DEFAULT_CONFIG_PATH as DEFAULT_CONFIG_PATH,
    HIDDEN_SETTINGS_KEYS as HIDDEN_SETTINGS_KEYS,
    READONLY_SETTINGS_KEYS as READONLY_SETTINGS_KEYS,
    USER_SETTINGS_KEYS as USER_SETTINGS_KEYS,
    Settings as Settings,
    get_writable_config_path as get_writable_config_path,
)


__all__ = [
    "CLI_SETTINGS_KEYS",
    "DEFAULT_CLI_SETTINGS",
    "DEFAULT_CONFIG_PATH",
    "HIDDEN_SETTINGS_KEYS",
    "READONLY_SETTINGS_KEYS",
    "USER_SETTINGS_KEYS",
    "Settings",
    "get_writable_config_path",
]
