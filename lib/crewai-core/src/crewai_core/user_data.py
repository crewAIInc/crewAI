"""Persistent per-user data + tracing-consent helpers.

This is the single source of truth for the ``.crewai_user.json`` file used by
both crewai (to record trace consent) and crewai-cli (to read/write it via
``crewai traces enable/disable/status``).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, cast

from crewai_core.lock_store import lock as store_lock
from crewai_core.paths import db_storage_path


logger = logging.getLogger(__name__)


def _user_data_file() -> Path:
    base = Path(db_storage_path())
    base.mkdir(parents=True, exist_ok=True)
    return base / ".crewai_user.json"


def _user_data_lock_name() -> str:
    """Return a stable lock name for the user data file."""
    return f"file:{os.path.realpath(_user_data_file())}"


def _load_user_data() -> dict[str, Any]:
    """Read the user-data JSON file, returning ``{}`` on missing/corrupt."""
    p = _user_data_file()
    if p.exists():
        try:
            return cast(dict[str, Any], json.loads(p.read_text()))
        except (json.JSONDecodeError, OSError, PermissionError) as e:
            logger.warning("Failed to load user data: %s", e)
    return {}


def _save_user_data(data: dict[str, Any]) -> None:
    """Write the full user-data dict, ignoring write errors with a warning."""
    try:
        p = _user_data_file()
        p.write_text(json.dumps(data, indent=2))
    except (OSError, PermissionError) as e:
        logger.warning("Failed to save user data: %s", e)


def update_user_data(updates: dict[str, Any]) -> None:
    """Atomically read-modify-write the user data file under a file lock.

    Args:
        updates: Key-value pairs to merge into the existing user data.
    """
    try:
        with store_lock(_user_data_lock_name()):
            data = _load_user_data()
            data.update(updates)
            _save_user_data(data)
    except (OSError, PermissionError) as e:
        logger.warning("Failed to update user data: %s", e)


def has_user_declined_tracing() -> bool:
    """Return True if the user has explicitly declined trace collection."""
    data = _load_user_data()
    if data.get("first_execution_done", False):
        return data.get("trace_consent", False) is False
    return False


def is_tracing_enabled() -> bool:
    """Return True if tracing should currently be active.

    Mirrors the runtime gate (``crewai.events.listeners.tracing.utils.
    should_enable_tracing``): ``CREWAI_TRACING_ENABLED=true`` always activates;
    otherwise recorded consent activates; otherwise off. Used by
    ``crewai traces status`` so the displayed state matches what crews and
    flows actually do.
    """
    if os.getenv("CREWAI_TRACING_ENABLED", "false").lower() in ("true", "1"):
        return True
    if has_user_declined_tracing():
        return False
    data = _load_user_data()
    return data.get("trace_consent", False) is not False
