"""
Utilities for creating visual representations of flow structures.

This module provides functions for generating network visualizations of flows,
including node placement, edge creation, and visual styling. It handles the
conversion of flow structures into visual network graphs with appropriate
styling and layout.

Example
-------
>>> flow = Flow()
>>> net = Network(directed=True)
>>> node_positions = compute_positions(flow, node_levels)
>>> add_nodes_to_network(net, flow, node_positions, node_styles)
>>> add_edges(net, flow, node_positions, colors)
"""

import ast
import inspect
from typing import Any, Dict, List, Tuple, Union

from .utils import (
    build_ancestor_dict,
    build_parent_children_dict,
    get_child_index,
    is_ancestor,
)


def method_calls_crew(method: Any) -> bool:
    """
    Check if the method contains a call to `.crew()`.

    Parameters
    ----------
    method : Any
        The method to analyze for crew() calls.

    Returns
    -------
    bool
        True if the method calls .crew(), False otherwise.

    Notes
    -----
    Uses AST analysis to detect method calls, specifically looking for
    attribute access of 'crew'.
    """
    try:
        source = inspect.getsource(method)
        source = inspect.cleandoc(source)
        tree = ast.parse(source)
    except Exception as e:
        print(f"Could not parse method {method.__name__}: {e}")
        return False

    class CrewCallVisitor(ast.NodeVisitor):
        """AST visitor to detect .crew() method calls."""
        def __init__(self):
            self.found = False

        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute):
                if node.func.attr == "crew":
                    self.found = True
            self.generic_visit(node)

    visitor = CrewCallVisitor()
    visitor.visit(tree)
    return visitor.found


def add_nodes_to_network(
    net: Any,
    flow: Any,
    node_positions: Dict[str, Tuple[float, float]],
    node_styles: Dict[str, Dict[str, Any]]
) -> None:
    """
    Add nodes to the network visualization with appropriate styling.

    Parameters
    ----------
    net : Any
        The pyvis Network instance to add nodes to.
    flow : Any
        The flow instance containing method information.
    node_positions : Dict[str, Tuple[float, float]]
        Dictionary mapping node names to their (x, y) positions.
    node_styles : Dict[str, Dict[str, Any]]
        Dictionary containing style configurations for different node types.

    Notes
    -----
    Node types include:
    - Start methods
    - Router methods
    - Crew methods
    - Regular methods
    """
    def human_friendly_label(method_name):
        return method_name.replace("_", " ").title()

    for method_name, (x, y) in node_positions.items():
        method = flow._methods.get(method_name)
        if hasattr(method, "__is_start_method__"):
            node_style = node_styles["start"]
        elif hasattr(method, "__is_router__"):
            node_style = node_styles["router"]
        elif method_calls_crew(method):
            node_style = node_styles["crew"]
        else:
            node_style = node_styles["method"]

        node_style = node_style.copy()
        label = human_friendly_label(method_name)

        node_style.update(
            {
                "label": label,
                "shape": "box",
                "font": {
                    "multi": "html",
                    "color": node_style.get("font", {}).get("color", "#FFFFFF"),
                },
            }
        )

        net.add_node(
            method_name,
            x=x,
            y=y,
            fixed=True,
            physics=False,
            **node_style,
        )


def compute_positions(
    flow: Any,
    node_levels: Dict[str, int],
    y_spacing: float = 150,
    x_spacing: float = 300
) -> Dict[str, Tuple[float, float]]:
    """
    Compute the (x, y) positions for each node in the flow graph.

    Parameters
    ----------
    flow : Any
        The flow instance to compute positions for.
    node_levels : Dict[str, int]
        Dictionary mapping node names to their hierarchical levels.
    y_spacing : float, optional
        Vertical spacing between levels, by default 150.
    x_spacing : float, optional
        Horizontal spacing between nodes, by default 300.

    Returns
    -------
    Dict[str, Tuple[float, float]]
        Dictionary mapping node names to their (x, y) coordinates.
    """
    level_nodes: Dict[int, List[str]] = {}
    node_positions: Dict[str, Tuple[float, float]] = {}

    for method_name, level in node_levels.items():
        level_nodes.setdefault(level, []).append(method_name)

    for level, nodes in level_nodes.items():
        x_offset = -(len(nodes) - 1) * x_spacing / 2  # Center nodes horizontally
        for i, method_name in enumerate(nodes):
            x = x_offset + i * x_spacing
            y = level * y_spacing
            node_positions[method_name] = (x, y)

    return node_positions


def add_edges(
    net: Any,
    flow: Any,
    node_positions: Dict[str, Tuple[float, float]],
    colors: Dict[str, str]
) -> None:
    edge_smooth: Dict[str, Union[str, float]] = {"type": "continuous"}  # Default value
    """
    Add edges to the network visualization with appropriate styling.

    Parameters
    ----------
    net : Any
        The pyvis Network instance to add edges to.
    flow : Any
        The flow instance containing edge information.
    node_positions : Dict[str, Tuple[float, float]]
        Dictionary mapping node names to their positions.
    colors : Dict[str, str]
        Dictionary mapping edge types to their colors.

    Notes
    -----
    - Handles both normal listener edges and router edges
    - Applies appropriate styling (color, dashes) based on edge type
    - Adds curvature to edges when needed (cycles or multiple children)
    """
    ancestors = build_ancestor_dict(flow)
    parent_children = build_parent_children_dict(flow)

    # Edges for normal listeners
    for method_name in flow._listeners:
        condition_type, trigger_methods = flow._listeners[method_name]
        is_and_condition = condition_type == "AND"

        for trigger in trigger_methods:
            # Check if nodes exist before adding edges
            if trigger in node_positions and method_name in node_positions:
                is_router_edge = any(
                    trigger in paths for paths in flow._router_paths.values()
                )
                edge_color = colors["router_edge"] if is_router_edge else colors["edge"]

                is_cycle_edge = is_ancestor(trigger, method_name, ancestors)
                parent_has_multiple_children = len(parent_children.get(trigger, [])) > 1
                needs_curvature = is_cycle_edge or parent_has_multiple_children

                if needs_curvature:
                    source_pos = node_positions.get(trigger)
                    target_pos = node_positions.get(method_name)

                    if source_pos and target_pos:
                        dx = target_pos[0] - source_pos[0]
                        smooth_type = "curvedCCW" if dx <= 0 else "curvedCW"
                        index = get_child_index(trigger, method_name, parent_children)
                        edge_smooth = {
                            "type": smooth_type,
                            "roundness": 0.2 + (0.1 * index),
                        }
                    else:
                        edge_smooth = {"type": "cubicBezier"}
                else:
                    edge_smooth.update({"type": "continuous"})

                edge_style = {
                    "color": edge_color,
                    "width": 2,
                    "arrows": "to",
                    "dashes": True if is_router_edge or is_and_condition else False,
                    "smooth": edge_smooth,
                }

                net.add_edge(trigger, method_name, **edge_style)
            else:
                # Nodes not found in node_positions. Check if it's a known router outcome and a known method.
                is_router_edge = any(
                    trigger in paths for paths in flow._router_paths.values()
                )
                # Check if method_name is a known method
                method_known = method_name in flow._methods

                # If it's a known router edge and the method is known, don't warn.
                # This means the path is legitimate, just not reflected as nodes here.
                if not (is_router_edge and method_known):
                    print(
                        f"Warning: No node found for '{trigger}' or '{method_name}'. Skipping edge."
                    )

    # Edges for router return paths
    for router_method_name, paths in flow._router_paths.items():
        for path in paths:
            for listener_name, (
                condition_type,
                trigger_methods,
            ) in flow._listeners.items():
                if path in trigger_methods:
                    if (
                        router_method_name in node_positions
                        and listener_name in node_positions
                    ):
                        is_cycle_edge = is_ancestor(
                            router_method_name, listener_name, ancestors
                        )
                        parent_has_multiple_children = (
                            len(parent_children.get(router_method_name, [])) > 1
                        )
                        needs_curvature = is_cycle_edge or parent_has_multiple_children

                        if needs_curvature:
                            source_pos = node_positions.get(router_method_name)
                            target_pos = node_positions.get(listener_name)

                            if source_pos and target_pos:
                                dx = target_pos[0] - source_pos[0]
                                smooth_type = "curvedCCW" if dx <= 0 else "curvedCW"
                                index = get_child_index(
                                    router_method_name, listener_name, parent_children
                                )
                                edge_smooth = {
                                    "type": smooth_type,
                                    "roundness": 0.2 + (0.1 * index),
                                }
                            else:
                                edge_smooth = {"type": "cubicBezier"}
                        else:
                            edge_smooth.update({"type": "continuous"})

                        edge_style = {
                            "color": colors["router_edge"],
                            "width": 2,
                            "arrows": "to",
                            "dashes": True,
                            "smooth": edge_smooth,
                        }
                        net.add_edge(router_method_name, listener_name, **edge_style)
                    else:
                        # Same check here: known router edge and known method?
                        method_known = listener_name in flow._methods
                        if not method_known:
                            print(
                                f"Warning: No node found for '{router_method_name}' or '{listener_name}'. Skipping edge."
                            )
