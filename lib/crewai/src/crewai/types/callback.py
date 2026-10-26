"""Serializable callback type for Pydantic models.

Provides a ``SerializableCallable`` type alias that enables full JSON
round-tripping of callback fields, e.g. ``"builtins.print"`` ↔ ``print``.
Lambdas, closures, bound methods, partials and callable instances serialize
to ``None``, so they do not survive a checkpoint restore — use module-level
named functions for checkpointable callbacks.
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


_TRUSTED_DESERIALIZE_VALUES = frozenset({"1", "true", "yes"})


def _trusted_deserialize() -> bool:
    """Return True only if ``CREWAI_DESERIALIZE_CALLBACKS`` is an explicit yes."""
    raw = os.environ.get("CREWAI_DESERIALIZE_CALLBACKS", "")
    return raw.strip().lower() in _TRUSTED_DESERIALIZE_VALUES


def string_to_callable(value: Any) -> Callable[..., Any]:
    """Convert a dotted path string to the callable it references.

    If *value* is already callable it is returned as-is, with a warning if
    :func:`callable_to_string` cannot serialize it.  Otherwise, it is treated as
    ``"module.qualname"`` and resolved via :func:`_resolve_dotted_path`.

    Args:
        value: A callable or a dotted-path string e.g. ``"builtins.print"``.

    Returns:
        The resolved callable.

    Raises:
        ValueError: If *value* is not callable or a resolvable dotted-path string.
    """
    if callable(value):
        if callable_to_string(value) is None:
            warnings.warn(
                f"A {type(value).__name__} callback cannot be saved in a "
                "checkpoint; it is dropped and will not run after a restore. "
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
    if not _trusted_deserialize():
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
    for i in range(len(parts), 0, -1):
        module_path = ".".join(parts[:i])
        try:
            obj: Any = importlib.import_module(module_path)
        except (ImportError, TypeError, ValueError):
            continue
        try:
            for attr in parts[i:]:
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        if callable(obj):
            return obj  # type: ignore[no-any-return]
    raise ValueError(f"Cannot resolve callback {path!r}")


def callable_to_string(fn: Callable[..., Any]) -> str | None:
    """Serialize a module-level callable as a ``"module.qualname"`` string.

    Args:
        fn: The callable to serialize.

    Returns:
        The dotted path, or ``None`` when :func:`string_to_callable` could not
        rebuild *fn* from one: lambdas, closures, bound instance methods,
        ``functools.partial`` objects and callable instances.
    """
    if inspect.ismethod(fn) and not inspect.isclass(fn.__self__):
        # The path would name the class attribute, which resolves to the
        # unbound function and fails without ``self`` when called.
        return None
    module = getattr(fn, "__module__", None)
    qualname = getattr(fn, "__qualname__", None)
    if module is None or qualname is None:
        return None
    if qualname.endswith("<lambda>") or "<locals>" in qualname:
        return None
    return f"{module}.{qualname}"


SerializableCallable = Annotated[
    Callable[..., Any],
    BeforeValidator(string_to_callable),
    PlainSerializer(callable_to_string, return_type=str, when_used="json"),
    WithJsonSchema({"type": "string"}),
]


def _instance_to_dotted_path(value: Any) -> str:
    """Serialize an instance to a dotted path naming its class."""
    if inspect.isclass(value):
        module = getattr(value, "__module__", "<unknown>")
        qualname = getattr(
            value, "__qualname__", getattr(value, "__name__", str(type(value)))
        )
        raise ValueError(f"Expected an instance, got class {module}.{qualname}.")
    cls = type(value)
    if cls.__module__ == "builtins":
        raise ValueError(
            f"Cannot serialize {value!r}: builtin values are not "
            "checkpointable instances."
        )
    module = getattr(cls, "__module__", None)
    qualname = getattr(cls, "__qualname__", None)
    if module is None or qualname is None:
        raise ValueError(
            f"Cannot serialize {value!r}: class missing __module__ or __qualname__. "
            "Use a module-level class for checkpointable instances."
        )
    if qualname.endswith("<lambda>") or "<locals>" in qualname:
        raise ValueError(
            f"Cannot serialize {value!r}: class defined in <locals>. "
            "Use a module-level class for checkpointable instances."
        )
    return f"{module}.{qualname}"


def _dotted_path_to_instance(value: Any) -> Any:
    """Resolve a dotted path to a class and instantiate it with no args.

    If *value* is already a non-string object it is returned as-is.
    """
    if value is None:
        return value
    if not isinstance(value, str):
        if inspect.isclass(value):
            raise ValueError(
                f"Expected an instance or dotted path string, got class "
                f"{getattr(value, '__module__', '<unknown>')}."
                f"{getattr(value, '__qualname__', getattr(value, '__name__', ''))}."
            )
        if type(value).__module__ == "builtins":
            raise ValueError(
                f"Expected an instance of a user-defined class or dotted "
                f"path string, got builtin value {value!r}."
            )
        return value
    if "." not in value:
        raise ValueError(
            f"Invalid provider path {value!r}: expected 'module.name' format"
        )
    if not _trusted_deserialize():
        raise ValueError(
            f"Refusing to resolve provider path {value!r}: "
            "set CREWAI_DESERIALIZE_CALLBACKS=1 to allow. "
            "Only enable this for trusted checkpoint data."
        )
    cls = _resolve_dotted_path(value)
    if not inspect.isclass(cls):
        raise ValueError(
            f"Invalid provider path {value!r}: expected a class, got "
            f"{type(cls).__name__}"
        )
    try:
        return cls()
    except TypeError as exc:
        raise ValueError(
            f"Cannot reinstantiate {value!r} with no arguments: {exc}. "
            "Only no-arg constructors are checkpointable; rebuild the "
            "instance manually and assign it after restore."
        ) from exc
