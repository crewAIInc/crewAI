"""Core utility functions for Flow class operations.

This module contains utility functions that are specifically designed to work
with the Flow class and require direct access to Flow class internals. These
utilities are separated from general-purpose utilities to maintain a clean
dependency structure and avoid circular imports.

Functions in this module are core to Flow functionality and are not related
to visualization or other optional features.
"""

import ast
import inspect
import textwrap
from typing import Any, Callable, Dict, List, Optional, Set, Union

from pydantic import BaseModel


def get_possible_return_constants(function: callable) -> Optional[List[str]]:
    """Extract possible string return values from a function by analyzing its source code.
    
    Analyzes the function's source code using AST to identify string constants that
    could be returned, including strings stored in dictionaries and direct returns.
    
    Args:
        function: The function to analyze for possible return values
        
    Returns:
        list[str] | None: List of possible string return values, or None if:
            - Source code cannot be retrieved
            - Source code has syntax/indentation errors
            - No string return values are found
            
    Raises:
        OSError: If source code cannot be retrieved
        IndentationError: If source code has invalid indentation
        SyntaxError: If source code has syntax errors
        
    Example:
        >>> def get_status():
        ...     paths = {"success": "completed", "error": "failed"}
        ...     return paths["success"]
        >>> get_possible_return_constants(get_status)
        ['completed', 'failed']
    """
    try:
        source = inspect.getsource(function)
    except OSError:
        # Can't get source code
        return None
    except Exception as e:
        print(f"Error retrieving source code for function {function.__name__}: {e}")
        return None

    try:
        # Remove leading indentation
        source = textwrap.dedent(source)
        # Parse the source code into an AST
        code_ast = ast.parse(source)
    except IndentationError as e:
        print(f"IndentationError while parsing source code of {function.__name__}: {e}")
        print(f"Source code:\n{source}")
        return None
    except SyntaxError as e:
        print(f"SyntaxError while parsing source code of {function.__name__}: {e}")
        print(f"Source code:\n{source}")
        return None
    except Exception as e:
        print(f"Unexpected error while parsing source code of {function.__name__}: {e}")
        print(f"Source code:\n{source}")
        return None

    return_values = set()
    dict_definitions = {}

    class DictionaryAssignmentVisitor(ast.NodeVisitor):
        def visit_Assign(self, node):
            # Check if this assignment is assigning a dictionary literal to a variable
            if isinstance(node.value, ast.Dict) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name):
                    var_name = target.id
                    dict_values = []
                    # Extract string values from the dictionary
                    for val in node.value.values:
                        if isinstance(val, ast.Constant) and isinstance(val.value, str):
                            dict_values.append(val.value)
                        # If non-string, skip or just ignore
                    if dict_values:
                        dict_definitions[var_name] = dict_values
            self.generic_visit(node)

    class ReturnVisitor(ast.NodeVisitor):
        def visit_Return(self, node):
            # Direct string return
            if isinstance(node.value, ast.Constant) and isinstance(
                node.value.value, str
            ):
                return_values.add(node.value.value)
            # Dictionary-based return, like return paths[result]
            elif isinstance(node.value, ast.Subscript):
                # Check if we're subscripting a known dictionary variable
                if isinstance(node.value.value, ast.Name):
                    var_name = node.value.value.id
                    if var_name in dict_definitions:
                        # Add all possible dictionary values
                        for v in dict_definitions[var_name]:
                            return_values.add(v)
            self.generic_visit(node)

    # First pass: identify dictionary assignments
    DictionaryAssignmentVisitor().visit(code_ast)
    # Second pass: identify returns
    ReturnVisitor().visit(code_ast)

    return list(return_values) if return_values else None


def is_ancestor(node: str, ancestor_candidate: str, ancestors: Dict[str, Set[str]]) -> bool:
    """Check if one node is an ancestor of another in the flow graph.
    
    Args:
        node: Target node to check ancestors for
        ancestor_candidate: Node to check if it's an ancestor
        ancestors: Dictionary mapping nodes to their ancestor sets
        
    Returns:
        bool: True if ancestor_candidate is an ancestor of node
        
    Raises:
        TypeError: If any argument has an invalid type
    """
    if not isinstance(node, str):
        raise TypeError("Argument 'node' must be a string")
    if not isinstance(ancestor_candidate, str):
        raise TypeError("Argument 'ancestor_candidate' must be a string")
    if not isinstance(ancestors, dict):
        raise TypeError("Argument 'ancestors' must be a dictionary")
    
    return ancestor_candidate in ancestors.get(node, set())
