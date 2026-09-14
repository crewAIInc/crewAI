"""Private typing helpers for the Python Flow DSL."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, TypeAlias, TypeVar

from crewai.flow.flow_wrappers import FlowCondition
from crewai.flow.types import FlowMethodCallable


__all__ = ["FlowMethodDecorator", "FlowTrigger"]

F = TypeVar("F", bound=Callable[..., Any])

FlowTrigger: TypeAlias = str | FlowMethodCallable[..., Any] | FlowCondition


class FlowMethodDecorator(Protocol):
    """Decorator returned by Flow DSL authoring helpers.

    The runtime wraps methods in FlowMethod subclasses, but the authoring
    contract preserves the decorated method's static callable type.
    """

    def __call__(self, func: F) -> F:
        raise NotImplementedError
