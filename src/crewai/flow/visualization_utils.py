import ast
import inspect

from .utils import (
    build_ancestor_dict,
    build_parent_children_dict,
    get_child_index,
    is_ancestor,
)


def method_calls_crew(method):
    """Check if the method calls `.crew()`."""
    try:
        source = inspect.getsource(method)
        source = inspect.cleandoc(source)
        tree = ast.parse(source)
    except Exception as e:
        print(f"Could not parse method {method.__name__}: {e}")
        return False

    class CrewCallVisitor(ast.NodeVisitor):
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


def add_nodes_to_network(net, flow, node_positions, node_styles):
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


def compute_positions(flow, node_levels, y_spacing=150, x_spacing=150):
    level_nodes = {}
    node_positions = {}

    for method_name, level in node_levels.items():
        level_nodes.setdefault(level, []).append(method_name)

    for level, nodes in level_nodes.items():
        x_offset = -(len(nodes) - 1) * x_spacing / 2  # Center nodes horizontally
        for i, method_name in enumerate(nodes):
            x = x_offset + i * x_spacing
            y = level * y_spacing
            node_positions[method_name] = (x, y)

    return node_positions


def add_edges(net, flow, node_positions, colors):
    ancestors = build_ancestor_dict(flow)
    parent_children = build_parent_children_dict(flow)

    for method_name in flow._listeners:
        condition_type, trigger_methods = flow._listeners[method_name]
        is_and_condition = condition_type == "AND"

        for trigger in trigger_methods:
            if trigger in flow._methods or trigger in flow._routers.values():
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
                    edge_smooth = False

                edge_style = {
                    "color": edge_color,
                    "width": 2,
                    "arrows": "to",
                    "dashes": True if is_router_edge or is_and_condition else False,
                    "smooth": edge_smooth,
                }

                net.add_edge(trigger, method_name, **edge_style)

    for router_method_name, paths in flow._router_paths.items():
        for path in paths:
            for listener_name, (
                condition_type,
                trigger_methods,
            ) in flow._listeners.items():
                if path in trigger_methods:
                    is_cycle_edge = is_ancestor(trigger, method_name, ancestors)
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
                        edge_smooth = False

                    edge_style = {
                        "color": colors["router_edge"],
                        "width": 2,
                        "arrows": "to",
                        "dashes": True,
                        "smooth": edge_smooth,
                    }
                    net.add_edge(router_method_name, listener_name, **edge_style)
