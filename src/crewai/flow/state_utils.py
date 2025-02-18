from datetime import date, datetime
from typing import Any

from pydantic import BaseModel

from crewai.flow import Flow


def export_state(flow: Flow) -> dict[str, Any]:
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
    return _to_serializable(flow._state)


def _to_serializable(obj: Any, max_depth: int = 5, _current_depth: int = 0) -> Any:
    if _current_depth >= max_depth:
        return repr(obj)

    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, (date, datetime)):
        return obj.isoformat()
    elif isinstance(obj, (list, tuple, set)):
        return [_to_serializable(item, max_depth, _current_depth + 1) for item in obj]
    elif isinstance(obj, dict):
        return {
            _to_serializable_key(key): _to_serializable(
                value, max_depth, _current_depth + 1
            )
            for key, value in obj.items()
        }
    elif isinstance(obj, BaseModel):
        return _to_serializable(obj.model_dump(), max_depth, _current_depth + 1)
    else:
        return repr(obj)


def _to_serializable_key(key: Any) -> str:
    if isinstance(key, (str, int)):
        return str(key)
    return f"key_{id(key)}_{repr(key)}"
