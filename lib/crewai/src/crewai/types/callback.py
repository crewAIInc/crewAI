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
import os
from typing import Annotated, Any
import warnings

from pydantic import BeforeValidator, WithJsonSchema
from pydantic.functional_serializers import PlainSerializer


def _is_non_roundtrippable(fn: object) -> bool:
    """Return ``True`` if *fn* cannot survive a serialize/deserialize round-trip.

    Built-in functions, plain module-level functions, and classes produce
    dotted paths that :func:`_resolve_dotted_path` can reliably resolve.
    Bound methods, ``functools.partial`` objects, callable class instances,
    lambdas, and closures all fail or silently change semantics during
    round-tripping.

    Args:
        fn: The object to check.

    Returns:
        ``True`` if *fn* would not round-trip through JSON serialization.
    """
    if inspect.isbuiltin(fn) or inspect.isclass(fn):
        return False
    if inspect.isfunction(fn):
        qualname = getattr(fn, "__qualname__", "")
        return qualname.endswith("<lambda>") or "<locals>" in qualname
    return True


def string_to_callable(value: Any) -> Callable[..., Any]:
    """Convert a dotted path string to the callable it references.

    If *value* is already callable it is returned as-is, with a warning if
    it cannot survive JSON round-tripping.  Otherwise, it is treated as
    ``"module.qualname"`` and resolved via :func:`_resolve_dotted_path`.

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
                f"{type(value).__name__} callbacks cannot be serialized "
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
    if "." not in value:
        raise ValueError(
            f"Invalid callback path {value!r}: expected 'module.name' format"
        )
    if not os.environ.get("CREWAI_DESERIALIZE_CALLBACKS"):
        raise ValueError(
            f"Refusing to resolve callback path {value!r}: "
            "set CREWAI_DESERIALIZE_CALLBACKS=1 to allow. "
            "Only enable this for trusted checkpoint data."
        )
    return _resolve_dotted_path(value)


def _resolve_dotted_path(path: str) -> Callable[..., Any]:
    """Import a module and walk attribute lookups to resolve a dotted path.

    Handles multi-level qualified names like ``"module.ClassName.method"``
    by trying progressively shorter module paths and resolving the remainder
    as chained attribute lookups.

    Args:
        path: A dotted string e.g. ``"builtins.print"`` or
              ``"mymodule.MyClass.my_method"``.

    Returns:
        The resolved callable.

    Raises:
        ValueError: If no valid module can be imported from the path.
    """
    parts = path.split(".")
    # Try importing progressively shorter prefixes as the module.
    for i in range(len(parts), 0, -1):
        module_path = ".".join(parts[:i])
        try:
            obj: Any = importlib.import_module(module_path)
        except (ImportError, TypeError, ValueError):
            continue
        # Walk the remaining attribute chain.
        try:
            for attr in parts[i:]:
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        if callable(obj):
            return obj  # type: ignore[no-any-return]
    raise ValueError(f"Cannot resolve callback {path!r}")


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
    module = getattr(fn, "__module__", None)
    qualname = getattr(fn, "__qualname__", None)
    if module is None or qualname is None:
        raise ValueError(
            f"Cannot serialize {fn!r}: missing __module__ or __qualname__. "
            "Use a module-level named function for checkpointable callbacks."
        )
    return f"{module}.{qualname}"


SerializableCallable = Annotated[
    Callable[..., Any],
    BeforeValidator(string_to_callable),
    PlainSerializer(callable_to_string, return_type=str, when_used="json"),
    WithJsonSchema({"type": "string"}),
]
