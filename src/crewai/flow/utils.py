"""General utility functions for flow execution.

This module has been deprecated. All functionality has been moved to:
- core_flow_utils.py: Core flow execution utilities
- flow_visual_utils.py: Visualization-related utilities

This module is kept as a temporary redirect to maintain backwards compatibility.
New code should import from the appropriate new modules directly.
"""

from typing import Any, Dict, List, Optional, Set

from .core_flow_utils import get_possible_return_constants, is_ancestor
from .flow_visual_utils import (
    build_ancestor_dict,
    build_parent_children_dict,
    calculate_node_levels,
    count_outgoing_edges,
    dfs_ancestors,
    get_child_index,
)

# Re-export all functions for backwards compatibility
__all__ = [
    'get_possible_return_constants',
    'calculate_node_levels',
    'count_outgoing_edges',
    'build_ancestor_dict',
    'dfs_ancestors',
    'is_ancestor',
    'build_parent_children_dict',
    'get_child_index',
]

# Function implementations have been moved to core_flow_utils.py and flow_visual_utils.py
