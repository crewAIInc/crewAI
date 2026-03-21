"""Schema loading utilities for vendored A2UI JSON schemas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_SCHEMA_DIR = Path(__file__).parent / "v0_8"

_SCHEMA_CACHE: dict[str, dict[str, Any]] = {}

SCHEMA_NAMES: frozenset[str] = frozenset(
    {
        "server_to_client",
        "client_to_server",
        "standard_catalog_definition",
        "server_to_client_with_standard_catalog",
    }
)


def load_schema(name: str) -> dict[str, Any]:
    """Load a vendored A2UI JSON schema by name.

    Args:
        name: Schema name without extension (e.g. ``"server_to_client"``).

    Returns:
        Parsed JSON schema dict.

    Raises:
        ValueError: If the schema name is not recognized.
        FileNotFoundError: If the schema file is missing from the package.
    """
    if name not in SCHEMA_NAMES:
        raise ValueError(f"Unknown schema {name!r}. Available: {sorted(SCHEMA_NAMES)}")

    if name in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[name]

    path = _SCHEMA_DIR / f"{name}.json"
    with path.open() as f:
        schema: dict[str, Any] = json.load(f)

    _SCHEMA_CACHE[name] = schema
    return schema
