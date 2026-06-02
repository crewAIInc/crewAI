"""Backwards-compatible shim. The implementation moved to ``crewai.flow.flow_definition``.

Import from ``crewai.flow.flow_definition`` directly in new code.
"""

from crewai.flow.flow_definition import (
    _extract_all_methods,
    _extract_all_methods_recursive,
    _extract_string_literals_from_type_annotation,
    _normalize_condition,
    _unwrap_function,
    build_ancestor_dict,
    build_parent_children_dict,
    calculate_node_levels,
    count_outgoing_edges,
    dfs_ancestors,
    extract_flow_definition,
    get_child_index,
    get_possible_return_constants,
    is_ancestor,
    is_flow_condition_dict,
    is_flow_condition_list,
    is_flow_method,
    is_flow_method_callable,
    is_flow_method_name,
    is_simple_flow_condition,
    process_router_paths,
)


__all__ = [
    "_extract_all_methods",
    "_extract_all_methods_recursive",
    "_extract_string_literals_from_type_annotation",
    "_normalize_condition",
    "_unwrap_function",
    "build_ancestor_dict",
    "build_parent_children_dict",
    "calculate_node_levels",
    "count_outgoing_edges",
    "dfs_ancestors",
    "extract_flow_definition",
    "get_child_index",
    "get_possible_return_constants",
    "is_ancestor",
    "is_flow_condition_dict",
    "is_flow_condition_list",
    "is_flow_method",
    "is_flow_method_callable",
    "is_flow_method_name",
    "is_simple_flow_condition",
    "process_router_paths",
]
