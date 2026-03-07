"""Serializable callable type for Pydantic models."""

from __future__ import annotations

from collections.abc import Callable
import importlib
from typing import Annotated, Any

from pydantic import BeforeValidator, PlainSerializer, WithJsonSchema


def _deserialize_callable(v: str | Callable[..., Any]) -> Callable[..., Any]:
    """Deserialize a dotted import path to a callable, or pass through if already callable."""
    if isinstance(v, str):
        module_path, _, name = v.rpartition(".")
        if not module_path:
            raise ValueError(f"Invalid callable path: {v!r} (expected 'module.name')")
        module = importlib.import_module(module_path)
        obj: Callable[..., Any] = getattr(module, name)
        if not callable(obj):
            raise ValueError(f"{v!r} resolved to {type(obj).__name__}, not a callable")
        return obj
    return v


def _serialize_callable(v: Callable[..., Any]) -> str:
    """Serialize a callable to its dotted import path."""
    module = getattr(v, "__module__", None)
    qualname = getattr(v, "__qualname__", None)
    name = getattr(v, "__name__", None)

    if not module or not name:
        raise ValueError(
            f"Cannot serialize {v!r}: missing __module__ or __name__. "
            "Only top-level named functions are serializable."
        )
    if qualname and "<" in qualname:
        raise ValueError(
            f"Cannot serialize {v!r}: lambdas and nested functions are not serializable. "
            "Use a top-level named function instead."
        )
    return f"{module}.{qualname or name}"


SerializableCallable = Annotated[
    Callable[..., Any],
    BeforeValidator(_deserialize_callable),
    PlainSerializer(_serialize_callable, return_type=str),
    WithJsonSchema({"type": "string"}),
]
