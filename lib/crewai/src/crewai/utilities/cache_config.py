"""Shared cache configuration helpers for Valkey/Redis URL parsing."""

from __future__ import annotations

import logging
import os
from typing import Any
from urllib.parse import urlparse


_logger = logging.getLogger(__name__)


def parse_cache_url() -> dict[str, Any] | None:
    """Parse VALKEY_URL or REDIS_URL from environment.

    Priority: VALKEY_URL > REDIS_URL.

    Returns:
        Dict with host, port, db, password keys, or None if no URL is set.
    """
    url = os.environ.get("VALKEY_URL") or os.environ.get("REDIS_URL")
    if not url:
        return None
    parsed = urlparse(url)
    return {
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 6379,
        "db": _parse_db_from_path(parsed.path),
        "password": parsed.password,
    }


def _parse_db_from_path(path: str | None) -> int:
    """Parse database number from URL path, defaulting to 0."""
    if not path or path == "/":
        return 0
    try:
        return int(path.lstrip("/"))
    except ValueError:
        _logger.warning(
            "Invalid database number in URL path: %s, using default 0", path
        )
        return 0


def get_aiocache_config() -> dict[str, Any]:
    """Build an aiocache configuration dict from environment.

    Uses VALKEY_URL or REDIS_URL (both are Redis-wire-compatible) to
    configure ``aiocache.RedisCache``.  Falls back to
    ``aiocache.SimpleMemoryCache`` when neither variable is set.

    Returns:
        Configuration dict suitable for ``aiocache.caches.set_config()``.
    """
    conn = parse_cache_url()
    if conn is not None:
        return {
            "default": {
                "cache": "aiocache.RedisCache",
                "endpoint": conn["host"],
                "port": conn["port"],
                "db": conn.get("db", 0),
                "password": conn.get("password"),
            }
        }
    return {
        "default": {
            "cache": "aiocache.SimpleMemoryCache",
        }
    }


def use_valkey_cache() -> bool:
    """Return True if VALKEY_URL is set in the environment."""
    return bool(os.environ.get("VALKEY_URL"))
