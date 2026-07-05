"""Backwards-compatible re-export surface for the Flow framework.

The implementation now lives in three modules, split by concern:

- ``crewai.flow.dsl`` -- authoring decorators (``@start`` / ``@listen`` /
  ``@router``, ``or_`` / ``and_``) and Python Flow class projection
- ``crewai.flow.flow_definition`` -- the serializable Flow Definition contract
- ``crewai.flow.runtime`` -- the Flow execution engine and state
- ``crewai.experimental.conversational_mixin`` -- experimental conversational
  runtime extension composed onto the public ``Flow`` class

Prefer importing from those modules in new code; this module preserves the
historical ``crewai.flow.flow`` import path.
"""

from typing import Any, TypeVar

from pydantic import BaseModel

from crewai.experimental.conversational_mixin import _ConversationalMixin
from crewai.flow.dsl import and_, listen, or_, router, start
from crewai.flow.runtime import (
    _INITIAL_STATE_CLASS_MARKER,
    Flow as RuntimeFlow,
    FlowMeta,
    FlowState,
)


T = TypeVar("T", bound=dict[str, Any] | BaseModel)


class Flow(_ConversationalMixin, RuntimeFlow[T]):
    """Public Flow class with experimental conversational extension behavior."""


__all__ = [
    "_INITIAL_STATE_CLASS_MARKER",
    "Flow",
    "FlowMeta",
    "FlowState",
    "and_",
    "listen",
    "or_",
    "router",
    "start",
]
