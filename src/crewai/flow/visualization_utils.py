import ast
import inspect
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, cast

from pyvis.network import Network

from crewai.flow.flow import Flow

from .core_flow_utils import is_ancestor
from .flow_visual_utils import (
    build_ancestor_dict,
    build_parent_children_dict,
    get_child_index,
)
from .path_utils import safe_path_join, validate_file_path


def method_calls_crew(method: Optional[Callable[..., Any]]) -> bool:
    """Check if the method contains a .crew() call in its implementation.
    
    Analyzes the method's source code using AST to detect if it makes any
    calls to the .crew() method, which indicates crew involvement in the
    flow execution.
    
    Args:
        method: The method to analyze for crew calls, can be None
        
    Returns:
        bool: True if the method contains a .crew() call, False otherwise
        
    Raises:
        TypeError: If input is not None and not a callable method
        ValueError: If method source code cannot be parsed
        RuntimeError: If unexpected error occurs during parsing
    """
    if method is None:
        return False
    if not callable(method):
        raise TypeError("Input must be a callable method")
        
    try:
        source = inspect.getsource(method)
        source = inspect.cleandoc(source)
        tree = ast.parse(source)
    except (TypeError, ValueError, OSError) as e:
        raise ValueError(f"Could not parse method {getattr(method, '__name__', str(method))}: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error parsing method: {e}")

    class CrewCallVisitor(ast.NodeVisitor):
        """AST visitor to detect .crew() method calls in source code.
        
        A specialized AST visitor that analyzes Python source code to precisely
        identify calls to the .crew() method, enabling accurate detection of
        crew involvement in flow methods.
        
        Attributes:
            found (bool): Indicates whether a .crew() call was found
        """
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


def add_nodes_to_network(net: Network, flow: Flow[Any], 
                     node_positions: Dict[str, Tuple[float, float]], 
                     node_styles: Dict[str, dict],
                     output_dir: Optional[str] = None) -> None:
    """Add nodes to the network visualization with precise styling and positioning.
    
    Creates and styles nodes in the visualization network based on their type
    (start, router, crew, or regular method) with fine-grained control over
    appearance and positioning.
    
    Args:
        net: The network visualization object to add nodes to
        flow: Flow object containing method definitions and relationships
        node_positions: Dictionary mapping method names to (x,y) coordinates
        node_styles: Dictionary mapping node types to their visual styles
        output_dir: Optional directory path for saving visualization assets
        
    Returns:
        None
        
    Raises:
        ValueError: If flow object is invalid or required styles are missing
        TypeError: If input arguments have incorrect types
        OSError: If output directory operations fail
        
    Note:
        Node styles are applied with precise control over shape, font, color,
        and positioning to ensure accurate visual representation of the flow.
        If output_dir is provided, it will be validated and created if needed.
    """
    if not hasattr(flow, '_methods'):
        raise ValueError("Invalid flow object: missing '_methods' attribute")
    if not isinstance(node_positions, dict):
        raise TypeError("node_positions must be a dictionary")
    if not isinstance(node_styles, dict):
        raise TypeError("node_styles must be a dictionary")
    
    required_styles = {'start', 'router', 'crew', 'method'}
    missing_styles = required_styles - set(node_styles.keys())
    if missing_styles:
        raise ValueError(f"Missing required node styles: {missing_styles}")
        
    # Validate and create output directory if specified
    if output_dir:
        try:
            output_dir = validate_file_path(output_dir, must_exist=False)
            os.makedirs(output_dir, exist_ok=True)
        except (ValueError, OSError) as e:
            raise OSError(f"Failed to create or validate output directory: {e}")
    def human_friendly_label(method_name: str) -> str: 
        """Convert method name to human-readable format.
        
        Args:
            method_name: Original method name with underscores
            
        Returns:
            str: Formatted method name with spaces and title case
        """
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

        # Handle file-based assets if output directory is provided
        if output_dir and node_style.get("image"):
            try:
                image_path = node_style["image"]
                safe_image_path = safe_path_join(output_dir, Path(image_path).name)
                node_style["image"] = str(safe_image_path)
            except (ValueError, OSError) as e:
                raise OSError(f"Failed to process node image path: {e}")

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


def compute_positions(flow: Flow[Any], node_levels: Dict[str, int], 
                   y_spacing: float = 150, x_spacing: float = 150) -> Dict[str, Tuple[float, float]]:
    """Calculate precise x,y coordinates for each node in the flow diagram.
    
    Computes optimal node positions with fine-grained control over spacing
    and alignment, ensuring clear visualization of flow hierarchy and
    relationships.
    
    Args:
        flow: Flow object containing method definitions
        node_levels: Dictionary mapping method names to their hierarchy levels
        y_spacing: Vertical spacing between hierarchy levels (default: 150)
        x_spacing: Horizontal spacing between nodes at same level (default: 150)
        
    Returns:
        dict[str, tuple[float, float]]: Dictionary mapping method names to
            their calculated (x,y) coordinates in the visualization
            
    Note:
        Positions are calculated to maintain clear hierarchical structure while
        ensuring optimal spacing and readability of the flow diagram.
    """
    if not hasattr(flow, '_methods'):
        raise ValueError("Invalid flow object: missing '_methods' attribute")
    if not isinstance(node_levels, dict):
        raise TypeError("node_levels must be a dictionary")
    if not isinstance(y_spacing, (int, float)) or y_spacing <= 0:
        raise ValueError("y_spacing must be a positive number")
    if not isinstance(x_spacing, (int, float)) or x_spacing <= 0:
        raise ValueError("x_spacing must be a positive number")
        
    if not node_levels:
        raise ValueError("node_levels dictionary cannot be empty")
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


def add_edges(net: Network, flow: Flow[Any], 
            node_positions: Dict[str, Tuple[float, float]], 
            colors: Dict[str, str],
            asset_dir: Optional[str] = None) -> None:
    if not hasattr(flow, '_methods'):
        raise ValueError("Invalid flow object: missing '_methods' attribute")
    if not hasattr(flow, '_listeners'):
        raise ValueError("Invalid flow object: missing '_listeners' attribute")
    if not hasattr(flow, '_router_paths'):
        raise ValueError("Invalid flow object: missing '_router_paths' attribute")
        
    if not isinstance(node_positions, dict):
        raise TypeError("node_positions must be a dictionary")
    if not isinstance(colors, dict):
        raise TypeError("colors must be a dictionary")
        
    required_colors = {'edge', 'router_edge'}
    missing_colors = required_colors - set(colors.keys())
    if missing_colors:
        raise ValueError(f"Missing required edge colors: {missing_colors}")
        
    # Validate asset directory if provided
    if asset_dir:
        try:
            asset_dir = validate_file_path(asset_dir, must_exist=False)
            os.makedirs(asset_dir, exist_ok=True)
        except (ValueError, OSError) as e:
            raise OSError(f"Failed to create or validate asset directory: {e}")
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
                        edge_config = {
                            "type": smooth_type,
                            "roundness": 0.2 + (0.1 * index),
                        }
                    else:
                        edge_config = {"type": "cubicBezier"}
                else:
                    edge_config = {"type": "straight"}

                edge_props: Dict[str, Any] = {
                    "color": edge_color,
                    "width": 2,
                    "arrows": "to",
                    "dashes": True if is_router_edge or is_and_condition else False,
                    "smooth": edge_config,
                }

                net.add_edge(trigger, method_name, **edge_props)
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
                                edge_config = {
                                    "type": smooth_type,
                                    "roundness": 0.2 + (0.1 * index),
                                }
                            else:
                                edge_config = {"type": "cubicBezier"}
                        else:
                            edge_config = {"type": "straight"}

                        router_edge_props: Dict[str, Any] = {
                            "color": colors["router_edge"],
                            "width": 2,
                            "arrows": "to",
                            "dashes": True,
                            "smooth": edge_config,
                        }
                        net.add_edge(router_method_name, listener_name, **router_edge_props)
                    else:
                        # Same check here: known router edge and known method?
                        method_known = listener_name in flow._methods
                        if not method_known:
                            print(
                                f"Warning: No node found for '{router_method_name}' or '{listener_name}'. Skipping edge."
                            )
