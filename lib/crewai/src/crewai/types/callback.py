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


def _is_lambda(fn: object) -> bool:
    """Return ``True`` if *fn* is a lambda expression.

    Uses ``__qualname__`` ending with ``"<lambda>"`` for resilience against
    ``__name__`` being reassigned.  ``inspect.isfunction`` gates the check
    so non-function callables (classes, partials, etc.) are never flagged.

    Args:
        fn: The object to check.

    Returns:
        ``True`` if *fn* is a lambda, ``False`` otherwise.
    """
    return inspect.isfunction(fn) and getattr(fn, "__qualname__", "").endswith(
        "<lambda>"
    )


def string_to_callable(value: Any) -> Callable[..., Any]:
    """Convert a dotted path string to the callable it references.

    If *value* is already callable it is returned as-is, with a warning if
    it is a lambda.  Otherwise, it is treated as ``"module.qualname"`` and
    resolved via :func:`importlib.import_module`.

    Args:
        value: A callable or a dotted-path string e.g. ``"builtins.print"``.

    Returns:
        The resolved callable.

    Raises:
        ModuleNotFoundError: If the module portion of the path cannot be imported.
        AttributeError: If the attribute cannot be found on the imported module.
    """
    if callable(value):
        if _is_lambda(value):
            warnings.warn(
                "Lambdas cannot be serialized and will prevent checkpointing. "
                "Use a module-level named function instead.",
                UserWarning,
                stacklevel=2,
            )
        return value  # type: ignore[no-any-return]
    module, func = value.rsplit(".", 1)
    return getattr(importlib.import_module(module), func)  # type: ignore[no-any-return]


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
    PlainSerializer(callable_to_string, return_type=str),
    WithJsonSchema({"type": "string"}),
]
