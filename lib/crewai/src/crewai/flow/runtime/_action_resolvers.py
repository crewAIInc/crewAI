from __future__ import annotations

from collections.abc import Callable
import importlib
from operator import attrgetter
from typing import TYPE_CHECKING, Any, cast

from crewai.flow.flow_definition import FlowActionDefinition


if TYPE_CHECKING:
    from crewai.flow.runtime import Flow


class InvalidActionRefError(ValueError):
    def __init__(self, ref: str) -> None:
        super().__init__(f"invalid callable {ref!r}; expected 'module:qualname'")


def _resolve_code_action(
    flow: Flow[Any], action: FlowActionDefinition
) -> Callable[..., Any]:
    ref = action.ref
    module_name, _, qualname = ref.partition(":")
    if "<" in ref or not module_name or not qualname:
        raise InvalidActionRefError(ref)
    try:
        target = attrgetter(qualname)(importlib.import_module(module_name))
    except (ImportError, AttributeError) as e:
        raise InvalidActionRefError(ref) from e
    if not callable(target):
        raise InvalidActionRefError(ref)
    handler = cast(Callable[..., Any], target)
    if getattr(handler, "__self__", None) is None:
        handler = handler.__get__(flow, type(flow))
    return handler


def resolve_action(flow: Flow[Any], action: FlowActionDefinition) -> Callable[..., Any]:
    """Turn one `do:` action into the callable the flow runs for that node."""
    if action.call == "code":
        return _resolve_code_action(flow, action)
    raise ValueError(f"unknown call type {action.call!r}")
