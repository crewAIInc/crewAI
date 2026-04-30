"""Schema loading utilities for vendored A2UI JSON schemas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_V08_DIR = Path(__file__).parent / "v0_8"
_V09_DIR = Path(__file__).parent / "v0_9"

_SCHEMA_CACHE: dict[str, dict[str, Any]] = {}

SCHEMA_NAMES: frozenset[str] = frozenset(
    {
        "server_to_client",
        "client_to_server",
        "standard_catalog_definition",
        "server_to_client_with_standard_catalog",
    }
)

V09_SCHEMA_NAMES: frozenset[str] = frozenset(
    {
        "server_to_client",
        "client_to_server",
        "common_types",
        "basic_catalog",
        "client_capabilities",
        "server_capabilities",
        "client_data_model",
    }
)


def load_schema(name: str, *, version: str = "v0.8") -> dict[str, Any]:
    """Load a vendored A2UI JSON schema by name and version.

    Args:
        name: Schema name without extension, e.g. ``"server_to_client"``.
        version: Protocol version, ``"v0.8"`` or ``"v0.9"``.

    Returns:
        Parsed JSON schema dict.

    Raises:
        ValueError: If the schema name or version is not recognized.
        FileNotFoundError: If the schema file is missing from the package.
    """
    if version == "v0.8":
        valid_names = SCHEMA_NAMES
        schema_dir = _V08_DIR
    elif version == "v0.9":
        valid_names = V09_SCHEMA_NAMES
        schema_dir = _V09_DIR
    else:
        raise ValueError(f"Unknown version {version!r}. Available: v0.8, v0.9")

    if name not in valid_names:
        raise ValueError(
            f"Unknown schema {name!r} for {version}. Available: {sorted(valid_names)}"
        )

    cache_key = f"{version}/{name}"
    if cache_key in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[cache_key]

    path = schema_dir / f"{name}.json"
    with path.open() as f:
        schema: dict[str, Any] = json.load(f)

    _SCHEMA_CACHE[cache_key] = schema
    return schema
