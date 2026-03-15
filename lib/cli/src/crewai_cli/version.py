"""Version utilities for CrewAI CLI."""

from collections.abc import Mapping
from datetime import datetime, timedelta
from functools import lru_cache
import importlib.metadata
import json
from pathlib import Path
from typing import Any
from urllib import request
from urllib.error import URLError

import appdirs
from packaging.version import InvalidVersion, Version, parse


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


def _find_latest_non_yanked_version(
    releases: Mapping[str, list[dict[str, Any]]],
) -> str | None:
    """Find the latest non-yanked version from PyPI releases data.

    Args:
        releases: PyPI releases dict mapping version strings to file info lists.

    Returns:
        The latest non-yanked version string, or None if all versions are yanked.
    """
    best_version: Version | None = None
    best_version_str: str | None = None

    for version_str, files in releases.items():
        try:
            v = parse(version_str)
        except InvalidVersion:
            continue

        if v.is_prerelease or v.is_devrelease:
            continue

        if not files:
            continue

        all_yanked = all(f.get("yanked", False) for f in files)
        if all_yanked:
            continue

        if best_version is None or v > best_version:
            best_version = v
            best_version_str = version_str

    return best_version_str


def _is_version_yanked(
    version_str: str,
    releases: Mapping[str, list[dict[str, Any]]],
) -> tuple[bool, str]:
    """Check if a specific version is yanked.

    Args:
        version_str: The version string to check.
        releases: PyPI releases dict mapping version strings to file info lists.

    Returns:
        Tuple of (is_yanked, yanked_reason).
    """
    files = releases.get(version_str, [])
    if not files:
        return False, ""

    all_yanked = all(f.get("yanked", False) for f in files)
    if not all_yanked:
        return False, ""

    for f in files:
        reason = f.get("yanked_reason", "")
        if reason:
            return True, str(reason)

    return True, ""


def get_latest_version_from_pypi(timeout: int = 2) -> str | None:
    """Get the latest non-yanked version of CrewAI from PyPI.

    Args:
        timeout: Request timeout in seconds.

    Returns:
        Latest non-yanked version string or None if unable to fetch.
    """
    cache_file = _get_cache_file()
    if cache_file.exists():
        try:
            cache_data = json.loads(cache_file.read_text())
            if _is_cache_valid(cache_data) and "current_version" in cache_data:
                version: str | None = cache_data.get("version")
                return version
        except (json.JSONDecodeError, OSError):
            pass

    try:
        with request.urlopen(
            "https://pypi.org/pypi/crewai/json", timeout=timeout
        ) as response:
            data = json.loads(response.read())
            releases: dict[str, list[dict[str, Any]]] = data["releases"]
            latest_version = _find_latest_non_yanked_version(releases)

            current_version = get_crewai_version()
            is_yanked, yanked_reason = _is_version_yanked(current_version, releases)

            cache_data = {
                "version": latest_version,
                "timestamp": datetime.now().isoformat(),
                "current_version": current_version,
                "current_version_yanked": is_yanked,
                "current_version_yanked_reason": yanked_reason,
            }
            cache_file.write_text(json.dumps(cache_data))

            return latest_version
    except (URLError, json.JSONDecodeError, KeyError, OSError):
        return None


def is_current_version_yanked() -> tuple[bool, str]:
    """Check if the currently installed version has been yanked on PyPI.

    Reads from cache if available, otherwise triggers a fetch.

    Returns:
        Tuple of (is_yanked, yanked_reason).
    """
    cache_file = _get_cache_file()
    if cache_file.exists():
        try:
            cache_data = json.loads(cache_file.read_text())
            if _is_cache_valid(cache_data) and "current_version" in cache_data:
                current = get_crewai_version()
                if cache_data.get("current_version") == current:
                    return (
                        bool(cache_data.get("current_version_yanked", False)),
                        str(cache_data.get("current_version_yanked_reason", "")),
                    )
        except (json.JSONDecodeError, OSError):
            pass

    get_latest_version_from_pypi()

    try:
        cache_data = json.loads(cache_file.read_text())
        return (
            bool(cache_data.get("current_version_yanked", False)),
            str(cache_data.get("current_version_yanked_reason", "")),
        )
    except (json.JSONDecodeError, OSError):
        return False, ""


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
