"""Type definitions for Flow structure visualization."""

from typing import Any

from typing_extensions import Required, TypedDict


__all__ = ["FlowStructure", "NodeMetadata", "StructureEdge"]


class NodeMetadata(TypedDict, total=False):
    """Metadata for a single node in the flow structure."""

    type: str
    is_router: bool
    router_events: list[str]
    condition_type: str | None
    trigger_condition_type: str | None
    trigger_methods: list[str]
    trigger_condition: dict[str, Any] | None
    class_name: str


class StructureEdge(TypedDict, total=False):
    """Represents a connection in the flow structure."""

    source: str
    target: str
    condition_type: str | None
    is_router_event: Required[bool]
    router_event: str | None


class FlowStructure(TypedDict):
    """Complete structure representation of a Flow."""

    nodes: dict[str, NodeMetadata]
    edges: list[StructureEdge]
    start_methods: list[str]
    router_methods: list[str]
