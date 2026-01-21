from __future__ import annotations

from datetime import date, datetime
import json
from typing import Any, TypeAlias
import uuid

from pydantic import BaseModel


SerializablePrimitive: TypeAlias = str | int | float | bool | None
Serializable: TypeAlias = (
    SerializablePrimitive | list["Serializable"] | dict[str, "Serializable"]
)


def to_serializable(
    obj: Any,
    exclude: set[str] | None = None,
    max_depth: int = 5,
    _current_depth: int = 0,
) -> Serializable:
    """Converts a Python object into a JSON-compatible representation.

    Supports primitives, datetime objects, collections, dictionaries, and
    Pydantic models. Recursion depth is limited to prevent infinite nesting.
    Non-convertible objects default to their string representations.

    Args:
        obj: Object to transform.
        exclude: Set of keys to exclude from the result.
        max_depth: Maximum recursion depth. Defaults to 5.
        _current_depth: Current recursion depth (for internal use).

    Returns:
        Serializable: A JSON-compatible structure.
    """
    if _current_depth >= max_depth:
        return repr(obj)

    if exclude is None:
        exclude = set()

    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if isinstance(obj, (list, tuple, set)):
        return [
            to_serializable(
                item, max_depth=max_depth, _current_depth=_current_depth + 1
            )
            for item in obj
        ]
    if isinstance(obj, dict):
        return {
            _to_serializable_key(key): to_serializable(
                obj=value,
                exclude=exclude,
                max_depth=max_depth,
                _current_depth=_current_depth + 1,
            )
            for key, value in obj.items()
            if key not in exclude
        }
    if isinstance(obj, BaseModel):
        try:
            return to_serializable(
                obj=obj.model_dump(exclude=exclude),
                max_depth=max_depth,
                _current_depth=_current_depth + 1,
            )
        except Exception:
            try:
                return {
                    _to_serializable_key(k): to_serializable(
                        v, max_depth=max_depth, _current_depth=_current_depth + 1
                    )
                    for k, v in obj.__dict__.items()
                    if k not in (exclude or set())
                }
            except Exception:
                return repr(obj)
    return repr(obj)


def _to_serializable_key(key: Any) -> str:
    if isinstance(key, (str, int)):
        return str(key)
    return f"key_{id(key)}_{key!r}"


def to_string(obj: Any) -> str | None:
    """Serializes an object into a JSON string.

    Args:
        obj: Object to serialize.

    Returns:
        A JSON-formatted string or `None` if empty.
    """
    serializable = to_serializable(obj)
    if serializable is None:
        return None
    return json.dumps(serializable)
