"""Resolve Python refs used in project definitions.

A ref must use this form: ``module:qualname``. ``module`` must name a Python
module we can import. ``qualname`` must name something inside that module. For
example, ``crewai_tools:SerperDevTool`` imports ``crewai_tools`` and returns
``SerperDevTool`` from it. Dots in ``qualname`` mean nested attributes.

Examples:

- ``crewai_tools:SerperDevTool`` imports ``crewai_tools`` and returns
  ``SerperDevTool``.
- ``my_app.tools:Factory.build`` imports ``my_app.tools``, gets ``Factory``,
  then gets ``build`` from ``Factory``.
- ``crewai_tools`` is invalid because it has no ``:``.
- ``crewai_tools:`` is invalid because it has no ``qualname``.

These helpers are the shared contract for YAML/JSON definitions:

- ``resolve_ref()`` checks the ref, imports the module, and returns the symbol
  as-is.
- ``resolve_class_ref()`` does the same work, then checks that the symbol is a
  class. It can also check that the class extends a base class. It does not
  create an object.

These helpers import user code. Code that must avoid that should check the raw
string shape instead.
"""

from __future__ import annotations

import importlib
import inspect
from operator import attrgetter
from typing import Any


class InvalidRefError(ValueError):
    """A definition ref that cannot be resolved to a live Python symbol."""


def resolve_ref(ref: str, *, field: str) -> Any:
    """Return the Python symbol named by a project definition field."""
    module_name, _, qualname = ref.partition(":")
    if "<" in ref or not module_name or not qualname:
        raise InvalidRefError(
            f"invalid {field} ref {ref!r}; expected 'module:qualname'"
        )
    try:
        return attrgetter(qualname)(importlib.import_module(module_name))
    except (ImportError, AttributeError) as e:
        raise InvalidRefError(f"unresolvable {field} ref {ref!r}") from e


def resolve_class_ref(
    ref: str,
    *,
    field: str,
    base_class: type[Any] | None = None,
) -> type[Any]:
    """Return the named class, with an optional base class check."""
    target = resolve_ref(ref, field=field)
    if not inspect.isclass(target):
        raise InvalidRefError(f"invalid {field} ref {ref!r}; expected a class")
    if base_class is not None and not issubclass(target, base_class):
        raise InvalidRefError(
            f"invalid {field} ref {ref!r}; expected a subclass of "
            f"{base_class.__module__}.{base_class.__name__}"
        )
    return target
