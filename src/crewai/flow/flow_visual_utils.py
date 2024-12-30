"""Utility functions for Flow visualization.

This module contains utility functions specifically designed for visualizing
Flow graphs and calculating layout information. These utilities are separated
from general-purpose utilities to maintain a clean dependency structure.
"""

from typing import TYPE_CHECKING, Dict, List, Set

if TYPE_CHECKING:
    from crewai.flow.flow import Flow


def calculate_node_levels(flow: Flow) -> Dict[str, int]:
    """Calculate the hierarchical level of each node in the flow graph.
    
    Uses breadth-first traversal to assign levels to nodes, starting with
    start methods at level 0. Handles both OR and AND conditions for listeners,
    and considers router paths when calculating levels.
    
    Args:
        flow: Flow instance containing methods, listeners, and router configurations
        
    Returns:
        dict[str, int]: Dictionary mapping method names to their hierarchical levels,
            where level 0 contains start methods and each subsequent level contains
            methods triggered by the previous level
            
    Example:
        >>> flow = Flow()
        >>> @flow.start
        ... def start(): pass
        >>> @flow.on("start")
        ... def second(): pass
        >>> calculate_node_levels(flow)
        {'start': 0, 'second': 1}
    """
    levels = {}
    queue = []
    visited = set()
    pending_and_listeners = {}

    # Make all start methods at level 0
    for method_name, method in flow._methods.items():
        if hasattr(method, "__is_start_method__"):
            levels[method_name] = 0
            queue.append(method_name)

    # Breadth-first traversal to assign levels
    while queue:
        current = queue.pop(0)
        current_level = levels[current]
        visited.add(current)

        for listener_name, (condition_type, trigger_methods) in flow._listeners.items():
            if condition_type == "OR":
                if current in trigger_methods:
                    if (
                        listener_name not in levels
                        or levels[listener_name] > current_level + 1
                    ):
                        levels[listener_name] = current_level + 1
                        if listener_name not in visited:
                            queue.append(listener_name)
            elif condition_type == "AND":
                if listener_name not in pending_and_listeners:
                    pending_and_listeners[listener_name] = set()
                if current in trigger_methods:
                    pending_and_listeners[listener_name].add(current)
                if set(trigger_methods) == pending_and_listeners[listener_name]:
                    if (
                        listener_name not in levels
                        or levels[listener_name] > current_level + 1
                    ):
                        levels[listener_name] = current_level + 1
                        if listener_name not in visited:
                            queue.append(listener_name)

        # Handle router connections
        if current in flow._routers:
            router_method_name = current
            paths = flow._router_paths.get(router_method_name, [])
            for path in paths:
                for listener_name, (
                    condition_type,
                    trigger_methods,
                ) in flow._listeners.items():
                    if path in trigger_methods:
                        if (
                            listener_name not in levels
                            or levels[listener_name] > current_level + 1
                        ):
                            levels[listener_name] = current_level + 1
                            if listener_name not in visited:
                                queue.append(listener_name)

    return levels


def count_outgoing_edges(flow: Flow) -> Dict[str, int]:
    """Count the number of outgoing edges for each node in the flow graph.
    
    An outgoing edge represents a connection from a method to a listener
    that it triggers. This is useful for visualization and analysis of
    flow structure.
    
    Args:
        flow: Flow instance containing methods and their connections
        
    Returns:
        dict[str, int]: Dictionary mapping method names to their number
            of outgoing connections
    """
    counts = {}
    for method_name in flow._methods:
        counts[method_name] = 0
    for method_name in flow._listeners:
        _, trigger_methods = flow._listeners[method_name]
        for trigger in trigger_methods:
            if trigger in flow._methods:
                counts[trigger] += 1
    return counts


def build_ancestor_dict(flow: Flow) -> Dict[str, Set[str]]:
    """Build a dictionary mapping each node to its set of ancestor nodes.
    
    Uses depth-first search to identify all ancestors (direct and indirect
    trigger methods) for each node in the flow graph. Handles both regular
    listeners and router paths.
    
    Args:
        flow: Flow instance containing methods and their relationships
        
    Returns:
        dict[str, set[str]]: Dictionary mapping each method name to a set
            of its ancestor method names
    """
    ancestors = {node: set() for node in flow._methods}
    visited = set()
    for node in flow._methods:
        if node not in visited:
            dfs_ancestors(node, ancestors, visited, flow)
    return ancestors




def dfs_ancestors(node: str, ancestors: Dict[str, Set[str]], 
                visited: Set[str], flow: Flow) -> None:
    """Perform depth-first search to populate the ancestors dictionary.
    
    Helper function for build_ancestor_dict that recursively traverses
    the flow graph to identify ancestors of each node.
    
    Args:
        node: Current node being processed
        ancestors: Dictionary mapping nodes to their ancestor sets
        visited: Set of already visited nodes
        flow: Flow instance containing the graph structure
    """
    if node in visited:
        return
    visited.add(node)

    # Handle regular listeners
    for listener_name, (_, trigger_methods) in flow._listeners.items():
        if node in trigger_methods:
            ancestors[listener_name].add(node)
            ancestors[listener_name].update(ancestors[node])
            dfs_ancestors(listener_name, ancestors, visited, flow)

    # Handle router methods separately
    if node in flow._routers:
        router_method_name = node
        paths = flow._router_paths.get(router_method_name, [])
        for path in paths:
            for listener_name, (_, trigger_methods) in flow._listeners.items():
                if path in trigger_methods:
                    # Only propagate the ancestors of the router method, not the router method itself
                    ancestors[listener_name].update(ancestors[node])
                    dfs_ancestors(listener_name, ancestors, visited, flow)


def build_parent_children_dict(flow: Flow) -> Dict[str, List[str]]:
    """Build a dictionary mapping each node to its list of child nodes.
    
    Maps both regular trigger methods to their listeners and router
    methods to their path listeners. Useful for visualization and
    traversal of the flow graph structure.
    
    Args:
        flow: Flow instance containing methods and their relationships
        
    Returns:
        dict[str, list[str]]: Dictionary mapping each method name to a
            sorted list of its child method names
    """
    parent_children = {}

    # Map listeners to their trigger methods
    for listener_name, (_, trigger_methods) in flow._listeners.items():
        for trigger in trigger_methods:
            if trigger not in parent_children:
                parent_children[trigger] = []
            if listener_name not in parent_children[trigger]:
                parent_children[trigger].append(listener_name)

    # Map router methods to their paths and to listeners
    for router_method_name, paths in flow._router_paths.items():
        for path in paths:
            # Map router method to listeners of each path
            for listener_name, (_, trigger_methods) in flow._listeners.items():
                if path in trigger_methods:
                    if router_method_name not in parent_children:
                        parent_children[router_method_name] = []
                    if listener_name not in parent_children[router_method_name]:
                        parent_children[router_method_name].append(listener_name)

    return parent_children


def get_child_index(parent: str, child: str, 
                 parent_children: Dict[str, List[str]]) -> int:
    """Get the index of a child node in its parent's sorted children list.
    
    Args:
        parent: Parent node name
        child: Child node name to find index for
        parent_children: Dictionary mapping parents to their children lists
        
    Returns:
        int: Zero-based index of the child in parent's sorted children list
        
    Raises:
        ValueError: If child is not found in parent's children list
    """
    children = parent_children.get(parent, [])
    children.sort()
    return children.index(child)
