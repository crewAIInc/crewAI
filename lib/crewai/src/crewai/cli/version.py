"""Version utilities for CrewAI CLI."""

from collections.abc import Mapping
from datetime import datetime, timedelta
from functools import lru_cache
import importlib.metadata
import json
from pathlib import Path
from typing import Any, cast
from urllib import request
from urllib.error import URLError

import appdirs
from packaging.version import InvalidVersion, parse


@lru_cache(maxsize=1)
def _get_cache_file() -> Path:
    """Get the path to the version cache file.

    Cached to avoid repeated filesystem operations.
    """
    cache_dir = Path(appdirs.user_cache_dir("crewai"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "version_cache.json"


def get_crewai_version() -> str:
    """Get the version number of CrewAI running the CLI."""
    return importlib.metadata.version("crewai")


def _is_cache_valid(cache_data: Mapping[str, Any]) -> bool:
    """Check if the cache is still valid, less than 24 hours old."""
    if "timestamp" not in cache_data:
        return False

    try:
        cache_time = datetime.fromisoformat(str(cache_data["timestamp"]))
        return datetime.now() - cache_time < timedelta(hours=24)
    except (ValueError, TypeError):
        return False


def get_latest_version_from_pypi(timeout: int = 2) -> str | None:
    """Get the latest version of CrewAI from PyPI.

    Args:
        timeout: Request timeout in seconds.

    Returns:
        Latest version string or None if unable to fetch.
    """
    cache_file = _get_cache_file()
    if cache_file.exists():
        try:
            cache_data = json.loads(cache_file.read_text())
            if _is_cache_valid(cache_data):
                return cast(str | None, cache_data.get("version"))
        except (json.JSONDecodeError, OSError):
            pass

    try:
        with request.urlopen(
            "https://pypi.org/pypi/crewai/json", timeout=timeout
        ) as response:
            data = json.loads(response.read())
            latest_version = cast(str, data["info"]["version"])

            cache_data = {
                "version": latest_version,
                "timestamp": datetime.now().isoformat(),
            }
            cache_file.write_text(json.dumps(cache_data))

            return latest_version
    except (URLError, json.JSONDecodeError, KeyError, OSError):
        return None


def check_version() -> tuple[str, str | None]:
    """Check current and latest versions.

    Returns:
        Tuple of (current_version, latest_version).
        latest_version is None if unable to fetch from PyPI.
    """
    current = get_crewai_version()
    latest = get_latest_version_from_pypi()
    return current, latest


def is_newer_version_available() -> tuple[bool, str, str | None]:
    """Check if a newer version is available.

    Returns:
        Tuple of (is_newer, current_version, latest_version).
    """
    current, latest = check_version()

    if latest is None:
        return False, current, None

    try:
        return parse(latest) > parse(current), current, latest
    except (InvalidVersion, TypeError):
        return False, current, latest
