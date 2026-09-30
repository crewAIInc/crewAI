"""Flow structure builder for definition-only Flow visualization."""

from __future__ import annotations

from collections import defaultdict
import logging
from typing import TYPE_CHECKING, Any, cast

from crewai.flow.constants import AND_CONDITION, OR_CONDITION
from crewai.flow.flow_definition import (
    FlowDefinition,
    FlowDefinitionCondition,
    FlowMethodDefinition,
)
from crewai.flow.visualization.types import FlowStructure, NodeMetadata, StructureEdge


logger = logging.getLogger(__name__)

__all__ = ["build_flow_structure", "calculate_execution_paths"]


if TYPE_CHECKING:
    from crewai.flow.flow import Flow


def _definition_condition_items(
    condition: dict[str, Any],
    key: str,
) -> list[FlowDefinitionCondition]:
    return cast(list[FlowDefinitionCondition], condition.get(key, []))


def _definition_condition_parts(
    condition: dict[str, Any],
) -> tuple[str, list[FlowDefinitionCondition]]:
    if "and" in condition:
        return AND_CONDITION, _definition_condition_items(condition, "and")
    return OR_CONDITION, _definition_condition_items(condition, "or")


def _condition_type_from_definition(
    condition: FlowDefinitionCondition | None,
) -> str | None:
    if isinstance(condition, dict):
        if "and" in condition:
            return AND_CONDITION
        if "or" in condition:
            return OR_CONDITION
    if isinstance(condition, str):
        return OR_CONDITION
    return None


def _runtime_condition_from_definition(
    condition: FlowDefinitionCondition,
) -> str | dict[str, Any]:
    if isinstance(condition, str):
        return condition
    condition_type, conditions = _definition_condition_parts(condition)
    return {
        "type": condition_type,
        "conditions": [_runtime_condition_from_definition(item) for item in conditions],
    }


def _method_trigger_condition(
    method_definition: FlowMethodDefinition,
) -> FlowDefinitionCondition | None:
    if method_definition.listen is not None:
        return method_definition.listen
    if isinstance(method_definition.start, str | dict):
        return method_definition.start
    return None


def _method_router_events(method_definition: FlowMethodDefinition) -> list[str]:
    if method_definition.human_feedback and method_definition.human_feedback.emit:
        return [str(event) for event in method_definition.human_feedback.emit]
    if method_definition.emit:
        return [str(event) for event in method_definition.emit]
    return []


def _extract_direct_or_triggers(
    condition: FlowDefinitionCondition,
) -> list[str]:
    if isinstance(condition, str):
        return [condition]
    condition_type, conditions = _definition_condition_parts(condition)
    if condition_type == AND_CONDITION:
        return []
    strings: list[str] = []
    for sub_condition in conditions:
        strings.extend(_extract_direct_or_triggers(sub_condition))
    return strings


def _extract_all_trigger_names(
    condition: FlowDefinitionCondition,
) -> list[str]:
    if isinstance(condition, str):
        return [condition]
    _, conditions = _definition_condition_parts(condition)
    strings: list[str] = []
    for sub_condition in conditions:
        strings.extend(_extract_all_trigger_names(sub_condition))
    return strings


def _create_edges_from_condition(
    condition: FlowDefinitionCondition,
    target: str,
    nodes: dict[str, NodeMetadata],
) -> list[StructureEdge]:
    edges: list[StructureEdge] = []

    if isinstance(condition, str):
        if condition in nodes:
            edges.append(
                StructureEdge(
                    source=condition,
                    target=target,
                    condition_type=OR_CONDITION,
                    is_router_event=False,
                )
            )
    elif isinstance(condition, dict):
        cond_type, conditions = _definition_condition_parts(condition)
        if cond_type == AND_CONDITION:
            triggers = _extract_all_trigger_names(condition)
            edges.extend(
                StructureEdge(
                    source=trigger,
                    target=target,
                    condition_type=AND_CONDITION,
                    is_router_event=False,
                )
                for trigger in triggers
                if trigger in nodes
            )
        else:
            for sub_condition in conditions:
                edges.extend(_create_edges_from_condition(sub_condition, target, nodes))

    return edges


def _flow_definition_from(
    flow_or_definition: Flow[Any] | type[Flow[Any]] | FlowDefinition,
) -> FlowDefinition:
    if isinstance(flow_or_definition, FlowDefinition):
        return flow_or_definition

    flow_class = (
        flow_or_definition
        if isinstance(flow_or_definition, type)
        else type(flow_or_definition)
    )
    flow_definition = getattr(flow_class, "flow_definition", None)
    if callable(flow_definition):
        return cast(FlowDefinition, flow_definition())
    raise TypeError(
        "build_flow_structure requires a FlowDefinition or a Flow class/instance "
        "with flow_definition()."
    )


def build_flow_structure(
    flow_or_definition: Flow[Any] | type[Flow[Any]] | FlowDefinition,
) -> FlowStructure:
    """Build a visualization structure projection from a FlowDefinition."""
    definition = _flow_definition_from(flow_or_definition)
    nodes: dict[str, NodeMetadata] = {}
    edges: list[StructureEdge] = []
    start_methods: list[str] = []
    router_methods: list[str] = []

    for method_name, method_definition in definition.methods.items():
        node_metadata: NodeMetadata = {"type": "listen", "class_name": definition.name}

        if method_definition.is_start:
            node_metadata["type"] = "start"
            start_methods.append(method_name)

        if method_definition.router:
            node_metadata["is_router"] = True
            node_metadata["type"] = "router"
            router_methods.append(method_name)
            router_events = _method_router_events(method_definition)
            if router_events:
                node_metadata["router_events"] = router_events

        trigger_condition = _method_trigger_condition(method_definition)
        condition_type = _condition_type_from_definition(trigger_condition)
        if condition_type is not None and trigger_condition is not None:
            node_metadata["trigger_condition_type"] = condition_type
            node_metadata["condition_type"] = condition_type
            extracted = _extract_all_trigger_names(trigger_condition)
            if extracted:
                node_metadata["trigger_methods"] = extracted
            runtime_condition = _runtime_condition_from_definition(trigger_condition)
            if isinstance(runtime_condition, dict):
                node_metadata["trigger_condition"] = runtime_condition

        if node_metadata.get("is_router") and "condition_type" not in node_metadata:
            node_metadata["condition_type"] = "IF"

        nodes[method_name] = node_metadata

    for method_name, method_definition in definition.methods.items():
        trigger_condition = _method_trigger_condition(method_definition)
        if trigger_condition is None:
            continue
        edges.extend(
            _create_edges_from_condition(trigger_condition, method_name, nodes)
        )

    all_string_triggers: set[str] = set()
    for method_definition in definition.methods.values():
        trigger_condition = _method_trigger_condition(method_definition)
        if trigger_condition is None:
            continue
        for trigger in _extract_direct_or_triggers(trigger_condition):
            if trigger not in nodes:
                all_string_triggers.add(trigger)

    all_router_events: set[str] = set()
    for router_method_name in router_methods:
        router_events = _method_router_events(definition.methods[router_method_name])
        if router_events and router_method_name in nodes:
            nodes[router_method_name]["router_events"] = router_events
            all_router_events.update(router_events)

        if not router_events:
            logger.warning(
                f"Router events for '{router_method_name}' are dynamic or not "
                f"statically inferable; static visualization may omit event edges."
            )

    orphaned_triggers = all_string_triggers - all_router_events
    if orphaned_triggers:
        logger.warning(
            f"Static visualization could not match listener triggers "
            f"{orphaned_triggers} to explicit router events. "
            f"Dynamic router values may still trigger these listeners at runtime."
        )

    for router_method_name in router_methods:
        router_events = _method_router_events(definition.methods[router_method_name])

        for event in router_events:
            for listener_name, method_definition in definition.methods.items():
                if listener_name == router_method_name:
                    continue

                trigger_condition = _method_trigger_condition(method_definition)
                if trigger_condition is None:
                    continue
                trigger_strings_from_cond = _extract_direct_or_triggers(
                    trigger_condition
                )

                if str(event) in trigger_strings_from_cond:
                    edges.append(
                        StructureEdge(
                            source=router_method_name,
                            target=listener_name,
                            condition_type=None,
                            is_router_event=True,
                            router_event=str(event),
                        )
                    )

    return FlowStructure(
        nodes=nodes,
        edges=edges,
        start_methods=start_methods,
        router_methods=router_methods,
    )


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
                "is_router": edge["is_router_event"],
                "condition": edge["condition_type"],
            }
        )

    all_nodes = set(structure["nodes"].keys())
    nodes_with_outgoing = set(edge["source"] for edge in structure["edges"])
    terminal_nodes = all_nodes - nodes_with_outgoing

    if not structure["start_methods"] or not terminal_nodes:
        return 0

    def count_paths_from(node: str, visited: set[str]) -> int:
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
