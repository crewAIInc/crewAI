"""Flow structure builder for analyzing Flow execution."""

from __future__ import annotations
from collections import defaultdict
import inspect
from typing import TYPE_CHECKING, Any

from crewai.flow.constants import OR_CONDITION, AND_CONDITION
from crewai.flow.flow_wrappers import FlowCondition
from crewai.flow.types import FlowMethodName
from crewai.flow.utils import (
    is_flow_condition_dict,
    is_simple_flow_condition,
)
from crewai.flow.visualization.schema import extract_method_signature
from crewai.flow.visualization.types import FlowStructure, NodeMetadata, StructureEdge


if TYPE_CHECKING:
    from crewai.flow.flow import Flow


def _extract_direct_or_triggers(
    condition: str | dict[str, Any] | list[Any] | FlowCondition,
) -> list[str]:
    """Extract direct OR-level trigger strings from a condition.

    This function extracts strings that would directly trigger a listener,
    meaning they appear at the top level of an OR condition. Strings nested
    inside AND conditions are NOT considered direct triggers for router paths.

    For example:
    - or_("a", "b") -> ["a", "b"] (both are direct triggers)
    - and_("a", "b") -> [] (neither are direct triggers, both required)
    - or_(and_("a", "b"), "c") -> ["c"] (only "c" is a direct trigger)

    Args:
        condition: Can be a string, dict, or list.

    Returns:
        List of direct OR-level trigger strings.
    """
    if isinstance(condition, str):
        return [condition]
    if isinstance(condition, dict):
        cond_type = condition.get("type", OR_CONDITION)
        conditions_list = condition.get("conditions", [])

        if cond_type == OR_CONDITION:
            strings = []
            for sub_cond in conditions_list:
                strings.extend(_extract_direct_or_triggers(sub_cond))
            return strings
        else:
            return []
    if isinstance(condition, list):
        strings = []
        for item in condition:
            strings.extend(_extract_direct_or_triggers(item))
        return strings
    if callable(condition) and hasattr(condition, "__name__"):
        return [condition.__name__]
    return []


def _extract_all_trigger_names(
    condition: str | dict[str, Any] | list[Any],
) -> list[str]:
    """Extract ALL trigger names from a condition for display purposes.

    Unlike _extract_direct_or_triggers, this extracts ALL strings and method
    names from the entire condition tree, including those nested in AND conditions.
    This is used for displaying trigger information in the UI.

    For example:
    - or_("a", "b") -> ["a", "b"]
    - and_("a", "b") -> ["a", "b"]
    - or_(and_("a", method_6), method_4) -> ["a", "method_6", "method_4"]

    Args:
        condition: Can be a string, dict, or list.

    Returns:
        List of all trigger names found in the condition.
    """
    if isinstance(condition, str):
        return [condition]
    if isinstance(condition, dict):
        conditions_list = condition.get("conditions", [])
        strings = []
        for sub_cond in conditions_list:
            strings.extend(_extract_all_trigger_names(sub_cond))
        return strings
    if isinstance(condition, list):
        strings = []
        for item in condition:
            strings.extend(_extract_all_trigger_names(item))
        return strings
    if callable(condition) and hasattr(condition, "__name__"):
        return [condition.__name__]
    return []


def _create_edges_from_condition(
    condition: str | dict[str, Any] | list[Any] | FlowCondition,
    target: str,
    nodes: dict[str, NodeMetadata],
) -> list[StructureEdge]:
    """Create edges from a condition tree, preserving AND/OR semantics.

    This function recursively processes the condition tree and creates edges
    with the appropriate condition_type for each trigger.

    For AND conditions, all triggers get edges with condition_type="AND".
    For OR conditions, triggers get edges with condition_type="OR".

    Args:
        condition: The condition tree (string, dict, or list).
        target: The target node name.
        nodes: Dictionary of all nodes for validation.

    Returns:
        List of StructureEdge objects representing the condition.
    """
    edges: list[StructureEdge] = []

    if isinstance(condition, str):
        if condition in nodes:
            edges.append(
                StructureEdge(
                    source=condition,
                    target=target,
                    condition_type=OR_CONDITION,
                    is_router_path=False,
                )
            )
    elif callable(condition) and hasattr(condition, "__name__"):
        method_name = condition.__name__
        if method_name in nodes:
            edges.append(
                StructureEdge(
                    source=method_name,
                    target=target,
                    condition_type=OR_CONDITION,
                    is_router_path=False,
                )
            )
    elif isinstance(condition, dict):
        cond_type = condition.get("type", OR_CONDITION)
        conditions_list = condition.get("conditions", [])

        if cond_type == AND_CONDITION:
            # For AND conditions, extract all triggers and create edges with AND type
            triggers = _extract_all_trigger_names(condition)
            for trigger in triggers:
                if trigger in nodes:
                    edges.append(
                        StructureEdge(
                            source=trigger,
                            target=target,
                            condition_type=AND_CONDITION,
                            is_router_path=False,
                        )
                    )
        else:  # OR condition
            # For OR conditions, recursively process sub-conditions
            for sub_cond in conditions_list:
                edges.extend(_create_edges_from_condition(sub_cond, target, nodes))
    elif isinstance(condition, list):
        # Process each item in the list
        for item in condition:
            edges.extend(_create_edges_from_condition(item, target, nodes))

    return edges


def build_flow_structure(flow: Flow[Any]) -> FlowStructure:
    """Build a structure representation of a Flow's execution.

    Args:
        flow: Flow instance to analyze.

    Returns:
        Dictionary with nodes, edges, start_methods, and router_methods.
    """
    nodes: dict[str, NodeMetadata] = {}
    edges: list[StructureEdge] = []
    start_methods: list[str] = []
    router_methods: list[str] = []

    for method_name, method in flow._methods.items():
        node_metadata: NodeMetadata = {"type": "listen"}

        if hasattr(method, "__is_start_method__") and method.__is_start_method__:
            node_metadata["type"] = "start"
            start_methods.append(method_name)

        if hasattr(method, "__is_router__") and method.__is_router__:
            node_metadata["is_router"] = True
            node_metadata["type"] = "router"
            router_methods.append(method_name)

            node_metadata["condition_type"] = "IF"

            if method_name in flow._router_paths:
                node_metadata["router_paths"] = [
                    str(p) for p in flow._router_paths[method_name]
                ]

        if hasattr(method, "__trigger_methods__") and method.__trigger_methods__:
            node_metadata["trigger_methods"] = [
                str(m) for m in method.__trigger_methods__
            ]

        if hasattr(method, "__condition_type__") and method.__condition_type__:
            if "condition_type" not in node_metadata:
                node_metadata["condition_type"] = method.__condition_type__

        if (
            hasattr(method, "__trigger_condition__")
            and method.__trigger_condition__ is not None
        ):
            node_metadata["trigger_condition"] = method.__trigger_condition__

            if "trigger_methods" not in node_metadata:
                extracted = _extract_all_trigger_names(method.__trigger_condition__)
                if extracted:
                    node_metadata["trigger_methods"] = extracted

        node_metadata["method_signature"] = extract_method_signature(
            method, method_name
        )

        try:
            source_code = inspect.getsource(method)
            node_metadata["source_code"] = source_code

            try:
                source_lines, start_line = inspect.getsourcelines(method)
                node_metadata["source_lines"] = source_lines
                node_metadata["source_start_line"] = start_line
            except (OSError, TypeError):
                pass

            try:
                source_file = inspect.getsourcefile(method)
                if source_file:
                    node_metadata["source_file"] = source_file
            except (OSError, TypeError):
                try:
                    class_file = inspect.getsourcefile(flow.__class__)
                    if class_file:
                        node_metadata["source_file"] = class_file
                except (OSError, TypeError):
                    pass
        except (OSError, TypeError):
            pass

        try:
            class_obj = flow.__class__

            if class_obj:
                class_name = class_obj.__name__

                bases = class_obj.__bases__
                if bases:
                    base_strs = []
                    for base in bases:
                        if hasattr(base, "__name__"):
                            if hasattr(base, "__origin__"):
                                base_strs.append(str(base))
                            else:
                                base_strs.append(base.__name__)
                        else:
                            base_strs.append(str(base))

                    try:
                        source_lines = inspect.getsource(class_obj).split("\n")
                        _, class_start_line = inspect.getsourcelines(class_obj)

                        for idx, line in enumerate(source_lines):
                            stripped = line.strip()
                            if stripped.startswith("class ") and class_name in stripped:
                                class_signature = stripped.rstrip(":")
                                node_metadata["class_signature"] = class_signature
                                node_metadata["class_line_number"] = (
                                    class_start_line + idx
                                )
                                break
                    except (OSError, TypeError):
                        class_signature = f"class {class_name}({', '.join(base_strs)})"
                        node_metadata["class_signature"] = class_signature
                else:
                    class_signature = f"class {class_name}"
                    node_metadata["class_signature"] = class_signature

                node_metadata["class_name"] = class_name
        except (OSError, TypeError, AttributeError):
            pass

        nodes[method_name] = node_metadata

    for listener_name, condition_data in flow._listeners.items():
        if is_simple_flow_condition(condition_data):
            cond_type, methods = condition_data
            edges.extend(
                StructureEdge(
                    source=str(trigger_method),
                    target=str(listener_name),
                    condition_type=cond_type,
                    is_router_path=False,
                )
                for trigger_method in methods
                if str(trigger_method) in nodes
            )
        elif is_flow_condition_dict(condition_data):
            edges.extend(
                _create_edges_from_condition(condition_data, str(listener_name), nodes)
            )

    for router_method_name in router_methods:
        if router_method_name not in flow._router_paths:
            continue

        router_paths = flow._router_paths[FlowMethodName(router_method_name)]

        for path in router_paths:
            for listener_name, condition_data in flow._listeners.items():
                trigger_strings_from_cond: list[str] = []

                if is_simple_flow_condition(condition_data):
                    _, methods = condition_data
                    trigger_strings_from_cond = [str(m) for m in methods]
                elif is_flow_condition_dict(condition_data):
                    trigger_strings_from_cond = _extract_direct_or_triggers(
                        condition_data
                    )

                if str(path) in trigger_strings_from_cond:
                    edges.append(
                        StructureEdge(
                            source=router_method_name,
                            target=str(listener_name),
                            condition_type=None,
                            is_router_path=True,
                        )
                    )

    for start_method in flow._start_methods:
        if start_method not in nodes and start_method in flow._methods:
            method = flow._methods[start_method]
            nodes[str(start_method)] = NodeMetadata(type="start")

            if hasattr(method, "__trigger_methods__") and method.__trigger_methods__:
                nodes[str(start_method)]["trigger_methods"] = [
                    str(m) for m in method.__trigger_methods__
                ]
            if hasattr(method, "__condition_type__") and method.__condition_type__:
                nodes[str(start_method)]["condition_type"] = method.__condition_type__

    return FlowStructure(
        nodes=nodes,
        edges=edges,
        start_methods=start_methods,
        router_methods=router_methods,
    )


def structure_to_dict(structure: FlowStructure) -> dict[str, Any]:
    """Convert FlowStructure to plain dictionary for serialization.

    Args:
        structure: FlowStructure to convert.

    Returns:
        Plain dictionary representation.
    """
    return {
        "nodes": dict(structure["nodes"]),
        "edges": list(structure["edges"]),
        "start_methods": list(structure["start_methods"]),
        "router_methods": list(structure["router_methods"]),
    }


def print_structure_summary(structure: FlowStructure) -> str:
    """Generate human-readable summary of Flow structure.

    Args:
        structure: FlowStructure to summarize.

    Returns:
        Formatted string summary.
    """
    lines: list[str] = [
        "Flow Execution Structure",
        "=" * 50,
        f"Total nodes: {len(structure['nodes'])}",
        f"Total edges: {len(structure['edges'])}",
        f"Start methods: {len(structure['start_methods'])}",
        f"Router methods: {len(structure['router_methods'])}",
        "",
    ]

    if structure["start_methods"]:
        lines.append("Start Methods:")
        for method_name in structure["start_methods"]:
            node = structure["nodes"][method_name]
            lines.append(f"  - {method_name}")
            if node.get("condition_type"):
                lines.append(f"    Condition: {node['condition_type']}")
            if node.get("trigger_methods"):
                lines.append(f"    Triggers on: {', '.join(node['trigger_methods'])}")
        lines.append("")

    if structure["router_methods"]:
        lines.append("Router Methods:")
        for method_name in structure["router_methods"]:
            node = structure["nodes"][method_name]
            lines.append(f"  - {method_name}")
            if node.get("router_paths"):
                lines.append(f"    Paths: {', '.join(node['router_paths'])}")
        lines.append("")

    if structure["edges"]:
        lines.append("Connections:")
        for edge in structure["edges"]:
            edge_type = ""
            if edge["is_router_path"]:
                edge_type = " [Router Path]"
            elif edge["condition_type"]:
                edge_type = f" [{edge['condition_type']}]"

            lines.append(f"  {edge['source']} -> {edge['target']}{edge_type}")
        lines.append("")

    return "\n".join(lines)


def calculate_execution_paths(structure: FlowStructure) -> int:
    """Calculate number of possible execution paths through the flow.

    Args:
        structure: FlowStructure to analyze.

    Returns:
        Number of possible execution paths.
    """
    graph = defaultdict(list)
    for edge in structure["edges"]:
        graph[edge["source"]].append(
            {
                "target": edge["target"],
                "is_router": edge["is_router_path"],
                "condition": edge["condition_type"],
            }
        )

    all_nodes = set(structure["nodes"].keys())
    nodes_with_outgoing = set(edge["source"] for edge in structure["edges"])
    terminal_nodes = all_nodes - nodes_with_outgoing

    if not structure["start_methods"] or not terminal_nodes:
        return 0

    def count_paths_from(node: str, visited: set[str]) -> int:
        """Recursively count execution paths from a given node.

        Args:
            node: Node name to start counting from.
            visited: Set of already visited nodes to prevent cycles.

        Returns:
            Number of execution paths from this node to terminal nodes.
        """
        if node in terminal_nodes:
            return 1

        if node in visited:
            return 0

        visited.add(node)

        outgoing = graph[node]
        if not outgoing:
            visited.remove(node)
            return 1

        if node in structure["router_methods"]:
            total = 0
            for edge_info in outgoing:
                target = str(edge_info["target"])
                total += count_paths_from(target, visited.copy())
            visited.remove(node)
            return total

        total = 0
        for edge_info in outgoing:
            target = str(edge_info["target"])
            total += count_paths_from(target, visited.copy())

        visited.remove(node)
        return total if total > 0 else 1

    total_paths = 0
    for start in structure["start_methods"]:
        total_paths += count_paths_from(start, set())

    return max(total_paths, 1)
