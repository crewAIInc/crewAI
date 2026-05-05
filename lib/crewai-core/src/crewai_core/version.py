"""Version utilities — installed version + PyPI freshness/yank checks.

Shared by both ``crewai`` and ``crewai-cli`` so the PyPI-checking logic lives
in one place. Frontends (``crewai version`` CLI, banner printer) consume the
helpers here without re-implementing them.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta
from functools import cache, lru_cache
import importlib.metadata
import json
from pathlib import Path
from typing import Any
from urllib import request
from urllib.error import URLError

import appdirs
from packaging.version import InvalidVersion, Version, parse


@cache
def get_crewai_version() -> str:
    """Return the installed crewAI version string.

    Falls back to ``"unknown"`` when neither crewai nor crewai-core are
    pip-installed (e.g. running directly from a source checkout).
    """
    try:
        return importlib.metadata.version("crewai")
    except importlib.metadata.PackageNotFoundError:
        pass
    try:
        return importlib.metadata.version("crewai-core")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


@lru_cache(maxsize=1)
def _get_cache_file() -> Path:
    """Return the path to the version cache file, creating the dir if needed."""
    cache_dir = Path(appdirs.user_cache_dir("crewai"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "version_cache.json"


def _is_cache_valid(cache_data: Mapping[str, Any]) -> bool:
    """Return True if the cache is less than 24 hours old."""
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
    """Return the latest non-prerelease, non-yanked version from PyPI releases."""
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
    """Return ``(yanked, reason)`` for ``version_str`` against PyPI releases."""
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
    """Return the latest non-yanked PyPI version of CrewAI, or ``None`` on failure."""
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
    """Return ``(yanked, reason)`` for the currently installed version."""
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
    """Return ``(current_version, latest_version)``; latest is ``None`` on fetch failure."""
    current = get_crewai_version()
    latest = get_latest_version_from_pypi()
    return current, latest


def is_newer_version_available() -> tuple[bool, str, str | None]:
    """Return ``(is_newer, current_version, latest_version)``."""
    current, latest = check_version()

    if latest is None:
        return False, current, None

    try:
        return parse(latest) > parse(current), current, latest
    except (InvalidVersion, TypeError):
        return False, current, latest
