"""Resolution of FlowDefinition refs (``module:qualname``) into live objects.

Every ref-shaped value in a definition — ``do`` actions, ``state.ref``,
``config.input_provider``, ``human_feedback.provider`` — resolves through
:func:`resolve_ref`. Failures are loud and name the field and the ref.
"""

from __future__ import annotations

from collections.abc import Callable
import importlib
import inspect
from operator import attrgetter
from typing import TYPE_CHECKING, Any, cast

from crewai.flow.flow_definition import FlowActionDefinition


if TYPE_CHECKING:
    from crewai.flow.runtime import Flow


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


def _resolve_code_action(
    flow: Flow[Any], action: FlowActionDefinition
) -> Callable[..., Any]:
    ref = action.ref
    target = resolve_ref(ref, field="do")
    if not callable(target):
        raise InvalidRefError(f"invalid do ref {ref!r}; object is not callable")
    handler = cast(Callable[..., Any], target)
    if getattr(handler, "__self__", None) is None:
        handler = handler.__get__(flow, type(flow))
    return handler


def resolve_action(flow: Flow[Any], action: FlowActionDefinition) -> Callable[..., Any]:
    """Turn one `do:` action into the callable the flow runs for that node."""
    if action.call == "code":
        return _resolve_code_action(flow, action)
    raise ValueError(f"unknown call type {action.call!r}")
