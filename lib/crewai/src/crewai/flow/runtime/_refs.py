"""Resolution of ``module:qualname`` refs into live Python objects."""

from __future__ import annotations

import importlib
import inspect
from operator import attrgetter
from typing import Any


class InvalidRefError(ValueError):
    """A definition ref that cannot be resolved to a live object."""


def resolve_ref(ref: str, *, field: str) -> Any:
    """Import the object a definition's `module:qualname` ref points to."""
    module_name, _, qualname = ref.partition(":")
    if "<" in ref or not module_name or not qualname:
        raise InvalidRefError(
            f"invalid {field} ref {ref!r}; expected 'module:qualname'"
        )
    try:
        return attrgetter(qualname)(importlib.import_module(module_name))
    except (ImportError, AttributeError) as e:
        raise InvalidRefError(f"unresolvable {field} ref {ref!r}") from e


def resolve_instance_ref(ref: str, *, field: str) -> Any:
    """Resolve a ref, auto-instantiating a no-arg class into an instance."""
    target = resolve_ref(ref, field=field)
    if not inspect.isclass(target):
        return target
    try:
        return target()
    except Exception as e:
        raise InvalidRefError(
            f"cannot instantiate {field} ref {ref!r} without arguments: {e}"
        ) from e
