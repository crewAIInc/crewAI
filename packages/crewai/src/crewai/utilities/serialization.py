import json
import uuid
from datetime import date, datetime
from typing import Any, Dict, List, Union

from pydantic import BaseModel

SerializablePrimitive = Union[str, int, float, bool, None]
Serializable = Union[
    SerializablePrimitive, List["Serializable"], Dict[str, "Serializable"]
]


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
        obj (Any): Object to transform.
        exclude (set[str], optional): Set of keys to exclude from the result.
        max_depth (int, optional): Maximum recursion depth. Defaults to 5.

    Returns:
        Serializable: A JSON-compatible structure.
    """
    if _current_depth >= max_depth:
        return repr(obj)

    if exclude is None:
        exclude = set()

    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, uuid.UUID):
        return str(obj)
    elif isinstance(obj, (date, datetime)):
        return obj.isoformat()
    elif isinstance(obj, (list, tuple, set)):
        return [
            to_serializable(
                item, max_depth=max_depth, _current_depth=_current_depth + 1
            )
            for item in obj
        ]
    elif isinstance(obj, dict):
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
    elif isinstance(obj, BaseModel):
        return to_serializable(
            obj=obj.model_dump(exclude=exclude),
            max_depth=max_depth,
            _current_depth=_current_depth + 1,
        )
    else:
        return repr(obj)


def _to_serializable_key(key: Any) -> str:
    if isinstance(key, (str, int)):
        return str(key)
    return f"key_{id(key)}_{repr(key)}"


def to_string(obj: Any) -> str | None:
    """Serializes an object into a JSON string.

    Args:
        obj (Any): Object to serialize.

    Returns:
        str | None: A JSON-formatted string or `None` if empty.
    """
    serializable = to_serializable(obj)
    if serializable is None:
        return None
    else:
        return json.dumps(serializable)
