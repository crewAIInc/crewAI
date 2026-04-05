"""Unified runtime state for crewAI.

``RuntimeState`` is a ``RootModel`` whose ``model_dump_json()`` produces a
complete, self-contained snapshot of every active entity in the program.

The ``Entity`` type alias and ``RuntimeState`` model are built at import time
in ``crewai/__init__.py`` after all forward references are resolved.
"""

from typing import Any


def _entity_discriminator(v: dict[str, Any] | object) -> str:
    if isinstance(v, dict):
        raw = v.get("entity_type", "agent")
    else:
        raw = getattr(v, "entity_type", "agent")
    return str(raw)
