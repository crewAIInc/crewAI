"""Type definitions for Flow structure visualization."""

from typing import Any, TypedDict


class NodeMetadata(TypedDict, total=False):
    """Metadata for a single node in the flow structure."""

    type: str
    is_router: bool
    router_paths: list[str]
    condition_type: str | None
    trigger_condition_type: str | None
    trigger_methods: list[str]
    trigger_condition: dict[str, Any] | None
    method_signature: dict[str, Any]
    source_code: str
    source_lines: list[str]
    source_start_line: int
    source_file: str
    class_signature: str
    class_name: str
    class_line_number: int


class StructureEdge(TypedDict, total=False):
    """Represents a connection in the flow structure."""

    source: str
    target: str
    condition_type: str | None
    is_router_path: bool
    router_path_label: str


class FlowStructure(TypedDict):
    """Complete structure representation of a Flow."""

    nodes: dict[str, NodeMetadata]
    edges: list[StructureEdge]
    start_methods: list[str]
    router_methods: list[str]
