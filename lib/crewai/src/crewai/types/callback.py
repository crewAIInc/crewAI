"""Serializable callback type for Pydantic models.

Provides a ``SerializableCallable`` type alias that enables full JSON
round-tripping of callback fields, e.g. ``"builtins.print"`` ↔ ``print``.
Lambdas and closures serialize to a dotted path but cannot be deserialized
back — use module-level named functions for checkpointable callbacks.
"""

from __future__ import annotations

from collections.abc import Callable
import importlib
import inspect
from typing import Annotated, Any
import warnings

from pydantic import BeforeValidator, WithJsonSchema
from pydantic.functional_serializers import PlainSerializer


def _is_non_roundtrippable(fn: object) -> bool:
    """Return ``True`` if *fn* cannot survive a serialize/deserialize round-trip.

    Detects lambdas via ``__qualname__`` ending with ``"<lambda>"`` and
    closures or nested functions via ``"<locals>"`` appearing anywhere in
    ``__qualname__``.  Both produce dotted paths that
    :func:`string_to_callable` cannot resolve back to the original object.
    ``inspect.isfunction`` gates the check so non-function callables
    like classes and partials are never flagged.

    Args:
        fn: The object to check.

    Returns:
        ``True`` if *fn* is a lambda, closure, or nested function.
    """
    if not inspect.isfunction(fn):
        return False
    qualname = getattr(fn, "__qualname__", "")
    return qualname.endswith("<lambda>") or "<locals>" in qualname


def string_to_callable(value: Any) -> Callable[..., Any]:
    """Convert a dotted path string to the callable it references.

    If *value* is already callable it is returned as-is, with a warning if
    it is a lambda, closure, or nested function.  Otherwise, it is treated
    as ``"module.qualname"`` and
    resolved via :func:`importlib.import_module`.

    Args:
        value: A callable or a dotted-path string e.g. ``"builtins.print"``.

    Returns:
        The resolved callable.

    Raises:
        ValueError: If *value* is not callable or a resolvable dotted-path string.
    """
    if callable(value):
        if _is_non_roundtrippable(value):
            warnings.warn(
                "Lambdas, closures, and nested functions cannot be serialized "
                "and will prevent checkpointing. "
                "Use a module-level named function instead.",
                UserWarning,
                stacklevel=2,
            )
        return value  # type: ignore[no-any-return]
    if not isinstance(value, str):
        raise ValueError(
            f"Expected a callable or dotted-path string, got {type(value).__name__}"
        )
    try:
        module, func = value.rsplit(".", 1)
    except ValueError:
        raise ValueError(
            f"Invalid callback path {value!r}: expected 'module.name' format"
        ) from None
    try:
        return getattr(importlib.import_module(module), func)  # type: ignore[no-any-return]
    except (ModuleNotFoundError, AttributeError) as exc:
        raise ValueError(f"Cannot resolve callback {value!r}: {exc}") from None


def callable_to_string(fn: Callable[..., Any]) -> str:
    """Serialize a callable to its dotted-path string representation.

    Uses ``fn.__module__`` and ``fn.__qualname__`` to produce a string such
    as ``"builtins.print"``.  Lambdas and closures produce paths that contain
    ``<locals>`` and cannot be round-tripped via :func:`string_to_callable`.

    Args:
        fn: The callable to serialize.

    Returns:
        A dotted string of the form ``"module.qualname"``.
    """
    return f"{fn.__module__}.{fn.__qualname__}"


SerializableCallable = Annotated[
    Callable[..., Any],
    BeforeValidator(string_to_callable),
    PlainSerializer(callable_to_string, return_type=str, when_used="json"),
    WithJsonSchema({"type": "string"}),
]
