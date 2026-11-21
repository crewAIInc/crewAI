"""Keep flow method output types across a checkpoint restore.

A checkpoint stores method outputs as JSON, so a Pydantic model or dataclass
comes back as a ``dict``. Its type is recorded next to it on write, and on
restore the ``dict`` is validated back into that type if the flow declares it.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, get_args, get_origin

from pydantic import BaseModel, TypeAdapter


logger = logging.getLogger(__name__)

_TYPE_KEYS = {"output": "output_type", "human_feedback": "human_feedback_type"}
# A HumanFeedbackResult keeps the reviewed method output in its own ``output``.
_FEEDBACK_OUTPUT_TYPE_KEYS = {
    "output": "feedback_output_type",
    "human_feedback": "human_feedback_output_type",
}


def _type_path(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def _add_allowed_types(annotation: Any, allowed: dict[str, Any]) -> None:
    """Add every class in ``annotation`` to ``allowed``, keyed by type path.

    A generic such as ``Box[Quote]`` is stored whole under ``Box``, so its
    items are rebuilt as ``Quote`` too.
    """
    origin = get_origin(annotation)
    if isinstance(annotation, type):
        allowed[_type_path(annotation)] = annotation
    elif isinstance(origin, type):
        allowed[_type_path(origin)] = annotation
    for arg in get_args(annotation):
        _add_allowed_types(arg, allowed)


def _is_structured(value: Any) -> bool:
    return isinstance(value, BaseModel) or dataclasses.is_dataclass(value)


def _holds_bare_model(value: Any) -> bool:
    """Whether ``value`` contains a bare ``BaseModel`` instance.

    Pydantic builds one for a field typed ``BaseModel``, such as
    ``CrewOutput.pydantic``. It holds no data and fails on attribute access.
    Validators can link objects into cycles, so each object is visited once.
    """
    pending: list[Any] = [value]
    seen: set[int] = set()
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        if type(item) is BaseModel:
            return True
        if isinstance(item, BaseModel):
            pending.extend(item.__dict__.values())
        elif dataclasses.is_dataclass(item):
            pending.extend(getattr(item, f.name) for f in dataclasses.fields(item))
        elif isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, (list, tuple, set)):
            pending.extend(item)
    return False


def record_output_types(entry: Any) -> Any:
    """Return a copy of a method output entry with its structured value types.

    Args:
        entry: An entry from ``Flow._method_outputs``.

    Returns:
        The entry with a ``module:qualname`` path for each Pydantic model or
        dataclass value, including the ``output`` of a ``HumanFeedbackResult``.
    """
    from crewai.flow.human_feedback import HumanFeedbackResult

    if not isinstance(entry, dict):
        return entry
    recorded = dict(entry)
    for key, type_key in _TYPE_KEYS.items():
        value = entry.get(key)
        if _is_structured(value):
            recorded[type_key] = _type_path(type(value))
        if isinstance(value, HumanFeedbackResult) and _is_structured(value.output):
            recorded[_FEEDBACK_OUTPUT_TYPE_KEYS[key]] = _type_path(type(value.output))
    return recorded


def rebuild_output_types(entry: dict[str, Any], declared: list[Any]) -> dict[str, Any]:
    """Return a copy of a restored entry with its recorded types rebuilt.

    A recorded type is rebuilt only if it appears in ``declared``, such as
    ``Quote`` in ``Quote | None`` or ``Box`` in ``Box[Quote]``, so a checkpoint
    cannot choose a type the flow does not declare. Any other value, or one
    that no longer validates, is kept as its JSON value.

    Args:
        entry: A method output entry read from a checkpoint.
        declared: The types and return annotations allowed for this entry.

    Returns:
        The entry without its type paths, with each allowed value rebuilt.
    """
    from crewai.flow.human_feedback import HumanFeedbackResult

    allowed: dict[str, Any] = {}
    for annotation in declared:
        _add_allowed_types(annotation, allowed)

    rebuilt = dict(entry)
    for key, type_key in _TYPE_KEYS.items():
        feedback_output_path = rebuilt.pop(_FEEDBACK_OUTPUT_TYPE_KEYS[key], None)
        if type_key not in rebuilt:
            continue
        path = rebuilt.pop(type_key)
        value = _rebuild(rebuilt[key], path, allowed)
        if feedback_output_path is not None and isinstance(value, HumanFeedbackResult):
            value.output = _rebuild(value.output, feedback_output_path, allowed)
        rebuilt[key] = value
    return rebuilt


def _rebuild(value: Any, path: str, allowed: dict[str, Any]) -> Any:
    """Return ``value`` validated into the allowed type at ``path``, or unchanged."""
    cls = allowed.get(path)
    if cls is None:
        logger.warning(
            "Method output type %r is not a return type the flow declares; "
            "keeping JSON. Annotate the method's return type to restore it.",
            path,
        )
        return value
    try:
        result = TypeAdapter(cls).validate_python(value)
        holds_bare_model = _holds_bare_model(result)
    except Exception:
        logger.warning(
            "Could not rebuild method output as %r; keeping JSON.",
            path,
            exc_info=True,
        )
        return value
    if holds_bare_model:
        logger.warning(
            "Method output %r would contain a bare BaseModel, whose data "
            "the checkpoint does not keep; keeping JSON.",
            path,
        )
        return value
    return result
