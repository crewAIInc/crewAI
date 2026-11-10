"""Type-safe serialization for flow method outputs in checkpoints.

Checkpoints store method outputs as JSON, so a method that returned a
Pydantic model or dataclass comes back from a restore as a plain dict.
These helpers record each output's type beside its value on checkpoint
writes and rebuild the original object on restores, so listeners of a
resumed flow receive the same input they got in the first run.
"""

from __future__ import annotations

import dataclasses
import importlib
import logging
import sys
from typing import Any

from pydantic import BaseModel, TypeAdapter

from crewai.types.callback import _trusted_deserialize


logger = logging.getLogger(__name__)

_TYPE_KEYS = (("output", "output_type"), ("human_feedback", "human_feedback_type"))


def structured_type_path(value: Any) -> str | None:
    """Return the import path of a structured value, or None.

    Args:
        value: The value whose type should be recorded.

    Returns:
        ``module.ClassName`` for Pydantic models and dataclass instances
        (the shapes a listener can read attributes from), otherwise None.
    """
    cls = type(value)
    if not isinstance(value, BaseModel) and not (
        dataclasses.is_dataclass(value) and not isinstance(value, type)
    ):
        return None
    module = getattr(cls, "__module__", None)
    qualname = getattr(cls, "__qualname__", None)
    if not module or not qualname:
        return None
    return f"{module}.{qualname}"


def annotate_method_output_types(entry: Any) -> Any:
    """Copy a method-output entry, adding type paths for structured values.

    Args:
        entry: One ``_method_outputs`` entry, a dict with ``method`` and
            ``output`` keys and an optional ``human_feedback`` key.

    Returns:
        A shallow copy with ``output_type`` / ``human_feedback_type`` set
        when the corresponding value is a Pydantic model or dataclass.
        Non-dict entries pass through unchanged.
    """
    if not (isinstance(entry, dict) and "output" in entry):
        return entry
    annotated = dict(entry)
    output_type = structured_type_path(entry["output"])
    if output_type:
        annotated["output_type"] = output_type
    feedback = entry.get("human_feedback")
    feedback_type = structured_type_path(feedback) if feedback is not None else None
    if feedback_type:
        annotated["human_feedback_type"] = feedback_type
    return annotated


def _resolve_type_path(type_path: str) -> Any:
    """Resolve a class recorded by :func:`structured_type_path`, or None.

    Tolerant by design: a checkpoint may reference a class that is no longer
    importable (a script's or notebook's locals, renamed code), and restore
    must degrade to the plain JSON value instead of failing the run.

    Resolution is limited to modules the running application already
    imported, so a checkpoint cannot make CrewAI import a module of its
    choosing: rebuilding a resumed flow's own models needs no import, because
    the module that defines the flow already imported them. Importing an
    unloaded path stays behind ``CREWAI_DESERIALIZE_CALLBACKS``, like every
    other dotted path read from a checkpoint.
    """
    parts = type_path.split(".")
    if not parts:
        return None
    trusted = _trusted_deserialize()
    for split in range(len(parts), 0, -1):
        name = ".".join(parts[:split])
        obj: Any = sys.modules.get(name)
        if obj is None:
            if not trusted:
                continue
            try:
                obj = importlib.import_module(name)
            except ImportError:
                continue
        try:
            for part in parts[split:]:
                obj = getattr(obj, part)
        except AttributeError:
            continue
        return obj
    return None


def _rebuild_value(cls: Any, value: Any) -> Any | None:
    """Validate a JSON value into an instance of its recorded type.

    A None return means the rebuild failed and the caller should keep the
    plain JSON value: restore must degrade, never fail the resumed run, so
    any validation error (including ones pydantic does not wrap) is caught.
    """
    try:
        if isinstance(cls, type) and issubclass(cls, BaseModel):
            return cls.model_validate(value)
        if dataclasses.is_dataclass(cls):
            return TypeAdapter(cls).validate_python(value)
    except Exception:
        logger.warning(
            "Could not rebuild %s from its checkpoint value",
            getattr(cls, "__qualname__", cls),
            exc_info=True,
        )
        return None
    return None


def rebuild_method_output_types(entry: Any) -> Any:
    """Rebuild structured outputs recorded with type paths on restore.

    Args:
        entry: One checkpointed method-output entry.

    Returns:
        The entry with ``output`` / ``human_feedback`` replaced by instances
        of their recorded types where possible; entries without type paths
        (legacy checkpoints, or types that could not be re-imported) pass
        through unchanged, keeping the plain JSON values.
    """
    if not isinstance(entry, dict):
        return entry
    rebuilt = entry
    for value_key, type_key in _TYPE_KEYS:
        type_path = entry.get(type_key)
        if not isinstance(type_path, str):
            continue
        cls = _resolve_type_path(type_path)
        if cls is None:
            logger.warning(
                "Could not import %s to rebuild the checkpointed %s",
                type_path,
                value_key,
            )
            continue
        value = _rebuild_value(cls, entry.get(value_key))
        if value is not None:
            rebuilt = {**rebuilt, value_key: value}
    return rebuilt
