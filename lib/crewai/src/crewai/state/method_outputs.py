"""Keep flow method output types across a checkpoint restore.

A checkpoint stores method outputs as JSON, so a Pydantic model or dataclass
comes back as a ``dict``. Its type is recorded next to it on write, and on
restore the ``dict`` is validated back into that type if the flow declares it.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, get_args

from pydantic import (
    BaseModel,
    PydanticSchemaGenerationError,
    TypeAdapter,
    ValidationError,
)


logger = logging.getLogger(__name__)

_TYPE_KEYS = {"output": "output_type", "human_feedback": "human_feedback_type"}


def _type_path(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


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
            recorded[type_key] = _type_path(type(value))
    return recorded


def rebuild_output_types(entry: dict[str, Any], declared: list[Any]) -> dict[str, Any]:
    """Return a copy of a restored entry with its recorded types rebuilt.

    A recorded type is rebuilt only if it is one of ``declared`` or a member of
    one, such as ``Quote`` in ``Quote | None``, so a checkpoint cannot choose
    a type the flow does not declare. Any other value, or one that no longer
    validates, is kept as its JSON value.

    Args:
        entry: A method output entry read from a checkpoint.
        declared: The types and return annotations the flow declares.

    Returns:
        The entry without its type paths, with each allowed value rebuilt.
    """
    allowed: dict[str, type] = {}
    for annotation in declared:
        for cls in get_args(annotation) or (annotation,):
            if isinstance(cls, type):
                allowed[_type_path(cls)] = cls

    rebuilt = dict(entry)
    for key, type_key in _TYPE_KEYS.items():
        if type_key not in rebuilt:
            continue
        path = rebuilt.pop(type_key)
        cls = allowed.get(path)
        if cls is None:
            logger.warning(
                "Method output type %r is not a return type the flow declares; "
                "keeping JSON. Annotate the method's return type to restore it.",
                path,
            )
            continue
        try:
            rebuilt[key] = TypeAdapter(cls).validate_python(rebuilt[key])
        except (PydanticSchemaGenerationError, ValidationError):
            logger.warning(
                "Method output no longer validates as %r; keeping JSON.",
                path,
                exc_info=True,
            )
    return rebuilt
