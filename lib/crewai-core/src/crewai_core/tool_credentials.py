"""Tool-repository credential helpers shared by crewai and crewai-cli."""

from __future__ import annotations

import os
from typing import Any

from crewai_core.project import read_toml
from crewai_core.settings import Settings


def build_env_with_tool_repository_credentials(
    repository_handle: str,
) -> dict[str, Any]:
    """Return a copy of ``os.environ`` augmented with UV_INDEX_* credentials
    for ``repository_handle``.

    The handle is normalized to upper-case with hyphens replaced by underscores
    (matching ``uv``'s env-var convention).
    """
    repository_handle = repository_handle.upper().replace("-", "_")
    settings = Settings()

    env = os.environ.copy()
    # Only override what the user actually saved. Writing "" for an unset
    # setting would erase credentials the caller already exported for this
    # index, which is common in CI.
    if settings.tool_repository_username is not None:
        env[f"UV_INDEX_{repository_handle}_USERNAME"] = str(
            settings.tool_repository_username
        )
    if settings.tool_repository_password is not None:
        env[f"UV_INDEX_{repository_handle}_PASSWORD"] = str(
            settings.tool_repository_password
        )

    return env


def build_env_with_all_tool_credentials() -> dict[str, Any]:
    """Return ``os.environ`` augmented with UV_INDEX_* credentials for every
    private index referenced under ``[tool.uv.sources]`` in ``pyproject.toml``.

    Errors reading ``pyproject.toml`` are swallowed — the un-augmented
    environment is returned in that case.
    """
    env = os.environ.copy()
    try:
        pyproject_data = read_toml()
        sources = pyproject_data.get("tool", {}).get("uv", {}).get("sources", {})

        for source_config in sources.values():
            if isinstance(source_config, dict):
                index = source_config.get("index")
                if index:
                    index_env = build_env_with_tool_repository_credentials(index)
                    env.update(index_env)
    except Exception:  # noqa: S110
        pass

    return env
