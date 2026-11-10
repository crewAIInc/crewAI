"""Keep flow method output types across a checkpoint restore.

A checkpoint stores method outputs as JSON, so a Pydantic model or dataclass
comes back as a ``dict``. Its type is recorded next to it on write and the
``dict`` is validated back into that type on restore. Types are only looked up
in modules that are already imported, so a restore never imports anything.
"""

from __future__ import annotations

import dataclasses
import logging
import sys
from typing import Any

from pydantic import BaseModel, TypeAdapter, ValidationError


logger = logging.getLogger(__name__)

_TYPE_KEYS = {"output": "output_type", "human_feedback": "human_feedback_type"}


def record_output_types(entry: Any) -> Any:
    """Return a copy of a method output entry with its structured value types.

    Args:
        entry: An entry from ``Flow._method_outputs``.

    Returns:
        The entry with a ``module:qualname`` path under ``output_type`` or
        ``human_feedback_type`` for each Pydantic model or dataclass value.
    """
    if not isinstance(entry, dict):
        return entry
    recorded = dict(entry)
    for key, type_key in _TYPE_KEYS.items():
        value = entry.get(key)
        if isinstance(value, BaseModel) or dataclasses.is_dataclass(value):
            cls = type(value)
            recorded[type_key] = f"{cls.__module__}:{cls.__qualname__}"
    return recorded


def rebuild_output_types(entry: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a restored entry with its recorded types rebuilt.

    A value whose type is not a Pydantic model or dataclass in a loaded
    module, or that no longer validates, is kept as its JSON value.

    Args:
        entry: A method output entry read from a checkpoint.

    Returns:
        The entry without its type paths, with each recorded value rebuilt.
    """
    rebuilt = dict(entry)
    for key, type_key in _TYPE_KEYS.items():
        if type_key in rebuilt:
            rebuilt[key] = _rebuild(rebuilt[key], rebuilt.pop(type_key))
    return rebuilt


def _rebuild(value: Any, path: str) -> Any:
    module_name, _, qualname = path.partition(":")
    cls: Any = sys.modules.get(module_name)
    for name in qualname.split("."):
        if cls is None:
            break
        # vars() rather than getattr() so a module __getattr__ never runs.
        cls = vars(cls).get(name)
    if not (
        isinstance(cls, type)
        and (issubclass(cls, BaseModel) or dataclasses.is_dataclass(cls))
    ):
        logger.warning("Cannot rebuild method output type %r; keeping JSON.", path)
        return value
    try:
        return TypeAdapter(cls).validate_python(value)
    except ValidationError:
        logger.warning(
            "Method output no longer validates as %r; keeping JSON.",
            path,
            exc_info=True,
        )
        return value
