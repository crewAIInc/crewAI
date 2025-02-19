import json
from datetime import date, datetime
from typing import Any, Dict, List, Union

from pydantic import BaseModel

from crewai.flow import Flow

SerializablePrimitive = Union[str, int, float, bool, None]
Serializable = Union[
    SerializablePrimitive, List["Serializable"], Dict[str, "Serializable"]
]


def export_state(flow: Flow) -> dict[str, Serializable]:
    """Exports the Flow's internal state as JSON-compatible data structures.

    Performs a one-way transformation of a Flow's state into basic Python types
    that can be safely serialized to JSON. To prevent infinite recursion with
    circular references, the conversion is limited to a depth of 5 levels.

    Args:
        flow: The Flow object whose state needs to be exported

    Returns:
        dict[str, Any]: The transformed state using JSON-compatible Python
            types.
    """
    result = to_serializable(flow._state)
    assert isinstance(result, dict)
    return result


def to_serializable(
    obj: Any, max_depth: int = 5, _current_depth: int = 0
) -> Serializable:
    """Converts a Python object into a JSON-compatible representation.

    Supports primitives, datetime objects, collections, dictionaries, and
    Pydantic models. Recursion depth is limited to prevent infinite nesting.
    Non-convertible objects default to their string representations.

    Args:
        obj (Any): Object to transform.
        max_depth (int, optional): Maximum recursion depth. Defaults to 5.

    Returns:
        Serializable: A JSON-compatible structure.
    """
    if _current_depth >= max_depth:
        return repr(obj)

    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, (date, datetime)):
        return obj.isoformat()
    elif isinstance(obj, (list, tuple, set)):
        return [to_serializable(item, max_depth, _current_depth + 1) for item in obj]
    elif isinstance(obj, dict):
        return {
            _to_serializable_key(key): to_serializable(
                value, max_depth, _current_depth + 1
            )
            for key, value in obj.items()
        }
    elif isinstance(obj, BaseModel):
        return to_serializable(obj.model_dump(), max_depth, _current_depth + 1)
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
