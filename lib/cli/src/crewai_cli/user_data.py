"""Standalone user-data helpers for the CLI package.

These mirror the functions in ``crewai.events.listeners.tracing.utils`` but
depend only on the standard library + *appdirs* so that crewai-cli can work
without importing the full crewai framework.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, cast

import appdirs


logger = logging.getLogger(__name__)


def _get_project_directory_name() -> str:
    return os.environ.get("CREWAI_STORAGE_DIR", Path.cwd().name)


def _db_storage_path() -> str:
    app_name = _get_project_directory_name()
    app_author = "CrewAI"
    data_dir = Path(appdirs.user_data_dir(app_name, app_author))
    data_dir.mkdir(parents=True, exist_ok=True)
    return str(data_dir)


def _user_data_file() -> Path:
    base = Path(_db_storage_path())
    base.mkdir(parents=True, exist_ok=True)
    return base / ".crewai_user.json"


def _load_user_data() -> dict[str, Any]:
    p = _user_data_file()
    if p.exists():
        try:
            return cast(dict[str, Any], json.loads(p.read_text()))
        except (json.JSONDecodeError, OSError, PermissionError) as e:
            logger.warning("Failed to load user data: %s", e)
    return {}


def _save_user_data(data: dict[str, Any]) -> None:
    try:
        p = _user_data_file()
        p.write_text(json.dumps(data, indent=2))
    except (OSError, PermissionError) as e:
        logger.warning("Failed to save user data: %s", e)


def is_tracing_enabled() -> bool:
    """Check if tracing is enabled (mirrors crewai core logic)."""
    data = _load_user_data()
    if (
        data.get("first_execution_done", False)
        and data.get("trace_consent", False) is False
    ):
        return False
    return os.getenv("CREWAI_TRACING_ENABLED", "false").lower() == "true"
