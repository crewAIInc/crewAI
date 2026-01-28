"""
Utility functions for flow visualization and dependency analysis.

This module provides core functionality for analyzing and manipulating flow structures,
including node level calculation, ancestor tracking, and return value analysis.
Functions in this module are primarily used by the visualization system to create
accurate and informative flow diagrams.

Example
-------
>>> flow = Flow()
>>> node_levels = calculate_node_levels(flow)
>>> ancestors = build_ancestor_dict(flow)
"""

from __future__ import annotations

import ast
from collections import defaultdict, deque
from enum import Enum
import inspect
import textwrap
from typing import TYPE_CHECKING, Any

from typing_extensions import TypeIs

from crewai.flow.constants import AND_CONDITION, OR_CONDITION
from crewai.flow.flow_wrappers import (
    FlowCondition,
    FlowConditions,
    FlowMethod,
    SimpleFlowCondition,
)
from crewai.flow.types import FlowMethodCallable, FlowMethodName
from crewai.utilities.printer import Printer


if TYPE_CHECKING:
    from crewai.flow.flow import Flow

_printer = Printer()


def _extract_string_literals_from_type_annotation(
    node: ast.expr,
    function_globals: dict[str, Any] | None = None,
) -> list[str]:
    """Extract string literals from a type annotation AST node.

    Handles:
    - Literal["a", "b", "c"]
    - "a" | "b" | "c" (union of string literals)
    - Just "a" (single string constant annotation)
    - Enum types with string values (e.g., class MyEnum(str, Enum))

    Args:
        node: The AST node representing a type annotation.
        function_globals: The globals dict from the function, used to resolve Enum types.

    Returns:
        List of string literals found in the annotation.
    """

    strings: list[str] = []

    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        strings.append(node.value)

    elif isinstance(node, ast.Name) and function_globals:
        enum_class = function_globals.get(node.id)
        if (
            enum_class is not None
            and isinstance(enum_class, type)
            and issubclass(enum_class, Enum)
        ):
            strings.extend(
                member.value for member in enum_class if isinstance(member.value, str)
            )

    elif isinstance(node, ast.Attribute) and function_globals:
        try:
            if isinstance(node.value, ast.Name):
                module = function_globals.get(node.value.id)
                if module is not None:
                    enum_class = getattr(module, node.attr, None)
                    if (
                        enum_class is not None
                        and isinstance(enum_class, type)
                        and issubclass(enum_class, Enum)
                    ):
                        strings.extend(
                            member.value
                            for member in enum_class
                            if isinstance(member.value, str)
                        )
        except (AttributeError, TypeError):
            pass

    elif isinstance(node, ast.Subscript):
        is_literal = False
        if isinstance(node.value, ast.Name) and node.value.id == "Literal":
            is_literal = True
        elif isinstance(node.value, ast.Attribute) and node.value.attr == "Literal":
            is_literal = True

        if is_literal:
            if isinstance(node.slice, ast.Tuple):
                strings.extend(
                    elt.value
                    for elt in node.slice.elts
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                )
            elif isinstance(node.slice, ast.Constant) and isinstance(
                node.slice.value, str
            ):
                strings.append(node.slice.value)

    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        strings.extend(
            _extract_string_literals_from_type_annotation(node.left, function_globals)
        )
        strings.extend(
            _extract_string_literals_from_type_annotation(node.right, function_globals)
        )

    return strings


def _unwrap_function(function: Any) -> Any:
    """Unwrap a function to get the original function with correct globals.

    Flow methods are wrapped by decorators like @router, @listen, etc.
    This function unwraps them to get the original function which has
    the correct __globals__ for resolving type annotations like Enums.

    Args:
        function: The potentially wrapped function.

    Returns:
        The unwrapped original function.
    """
    if hasattr(function, "__func__"):
        function = function.__func__

    if hasattr(function, "__wrapped__"):
        wrapped = function.__wrapped__
        if hasattr(wrapped, "unwrap"):
            return wrapped.unwrap()
        return wrapped

    return function


def get_possible_return_constants(
    function: Any, verbose: bool = True
) -> list[str] | None:
    """Extract possible string return values from a function using AST parsing.

    This function analyzes the source code of a router method to identify
    all possible string values it might return. It handles:
    - Return type annotations: -> Literal["a", "b"] or -> "a" | "b" | "c"
    - Enum type annotations: -> MyEnum (extracts string values from members)
    - Direct string literals: return "value"
    - Variable assignments: x = "value"; return x
    - Dictionary lookups: d = {"k": "v"}; return d[key]
    - Conditional returns: return "a" if cond else "b"
    - State attributes: return self.state.attr (infers from class context)

    Args:
        function: The function to analyze.

    Returns:
        List of possible string return values, or None if analysis fails.
    """
    unwrapped = _unwrap_function(function)

    try:
        source = inspect.getsource(function)
    except OSError:
        # Can't get source code
        return None
    except Exception as e:
        if verbose:
            _printer.print(
                f"Error retrieving source code for function {function.__name__}: {e}",
                color="red",
            )
        return None

    try:
        # Remove leading indentation
        source = textwrap.dedent(source)
        # Parse the source code into an AST
        code_ast = ast.parse(source)
    except IndentationError as e:
        if verbose:
            _printer.print(
                f"IndentationError while parsing source code of {function.__name__}: {e}",
                color="red",
            )
            _printer.print(f"Source code:\n{source}", color="yellow")
        return None
    except SyntaxError as e:
        if verbose:
            _printer.print(
                f"SyntaxError while parsing source code of {function.__name__}: {e}",
                color="red",
            )
            _printer.print(f"Source code:\n{source}", color="yellow")
        return None
    except Exception as e:
        if verbose:
            _printer.print(
                f"Unexpected error while parsing source code of {function.__name__}: {e}",
                color="red",
            )
            _printer.print(f"Source code:\n{source}", color="yellow")
        return None

    return_values: set[str] = set()

    function_globals = getattr(unwrapped, "__globals__", None)

    for node in ast.walk(code_ast):
        if isinstance(node, ast.FunctionDef):
            if node.returns:
                annotation_values = _extract_string_literals_from_type_annotation(
                    node.returns, function_globals
                )
                return_values.update(annotation_values)
            break  # Only process the first function definition
    dict_definitions: dict[str, list[str]] = {}
    variable_values: dict[str, list[str]] = {}
    state_attribute_values: dict[str, list[str]] = {}

    def extract_string_constants(node: ast.expr) -> list[str]:
        """Recursively extract all string constants from an AST node."""
        strings: list[str] = []
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            strings.append(node.value)
        elif isinstance(node, ast.IfExp):
            strings.extend(extract_string_constants(node.body))
            strings.extend(extract_string_constants(node.orelse))
        elif isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and len(node.args) >= 2
            ):
                default_arg = node.args[1]
                if isinstance(default_arg, ast.Constant) and isinstance(
                    default_arg.value, str
                ):
                    strings.append(default_arg.value)
        return strings

    class VariableAssignmentVisitor(ast.NodeVisitor):
        def visit_Assign(self, node: ast.Assign) -> None:
            # Check if this assignment is assigning a dictionary literal to a variable
            if isinstance(node.value, ast.Dict) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name):
                    var_name = target.id
                    # Extract string values from the dictionary
                    dict_values = [
                        val.value
                        for val in node.value.values
                        if isinstance(val, ast.Constant) and isinstance(val.value, str)
                    ]
                    if dict_values:
                        dict_definitions[var_name] = dict_values

            if len(node.targets) == 1:
                target = node.targets[0]
                var_name_alt: str | None = None
                if isinstance(target, ast.Name):
                    var_name_alt = target.id
                elif isinstance(target, ast.Attribute):
                    var_name_alt = f"{target.value.id if isinstance(target.value, ast.Name) else '_'}.{target.attr}"

                if var_name_alt:
                    strings = extract_string_constants(node.value)
                    if strings:
                        variable_values[var_name_alt] = strings

            self.generic_visit(node)

    def get_attribute_chain(node: ast.expr) -> str | None:
        """Extract the full attribute chain from an AST node.

        Examples:
            self.state.run_type -> "self.state.run_type"
            x.y.z -> "x.y.z"
            simple_var -> "simple_var"
        """
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = get_attribute_chain(node.value)
            if base:
                return f"{base}.{node.attr}"
        return None

    class ReturnVisitor(ast.NodeVisitor):
        def visit_Return(self, node: ast.Return) -> None:
            if (
                node.value
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                return_values.add(node.value.value)
            elif node.value and isinstance(node.value, ast.Subscript):
                if isinstance(node.value.value, ast.Name):
                    var_name_dict = node.value.value.id
                    if var_name_dict in dict_definitions:
                        for v in dict_definitions[var_name_dict]:
                            return_values.add(v)
            elif node.value:
                var_name_ret = get_attribute_chain(node.value)

                if var_name_ret and var_name_ret in variable_values:
                    for v in variable_values[var_name_ret]:
                        return_values.add(v)
                elif var_name_ret and var_name_ret in state_attribute_values:
                    for v in state_attribute_values[var_name_ret]:
                        return_values.add(v)

            self.generic_visit(node)

        def visit_If(self, node: ast.If) -> None:
            self.generic_visit(node)

    # Try to get the class context to infer state attribute values
    try:
        if hasattr(function, "__self__"):
            # Method is bound, get the class
            class_obj = function.__self__.__class__
        elif hasattr(function, "__qualname__") and "." in function.__qualname__:
            # Method is unbound but we can try to get class from module
            class_name = function.__qualname__.rsplit(".", 1)[0]
            if hasattr(function, "__globals__"):
                class_obj = function.__globals__.get(class_name)
            else:
                class_obj = None
        else:
            class_obj = None

        if class_obj is not None:
            try:
                class_source = inspect.getsource(class_obj)
                class_source = textwrap.dedent(class_source)
                class_ast = ast.parse(class_source)

                # Look for comparisons and assignments involving state attributes
                class StateAttributeVisitor(ast.NodeVisitor):
                    def visit_Compare(self, node: ast.Compare) -> None:
                        """Find comparisons like: self.state.attr == "value" """
                        left_attr = get_attribute_chain(node.left)

                        if left_attr:
                            for comparator in node.comparators:
                                if isinstance(comparator, ast.Constant) and isinstance(
                                    comparator.value, str
                                ):
                                    if left_attr not in state_attribute_values:
                                        state_attribute_values[left_attr] = []
                                    if (
                                        comparator.value
                                        not in state_attribute_values[left_attr]
                                    ):
                                        state_attribute_values[left_attr].append(
                                            comparator.value
                                        )

                        # Also check right side
                        for comparator in node.comparators:
                            right_attr = get_attribute_chain(comparator)
                            if (
                                right_attr
                                and isinstance(node.left, ast.Constant)
                                and isinstance(node.left.value, str)
                            ):
                                if right_attr not in state_attribute_values:
                                    state_attribute_values[right_attr] = []
                                if (
                                    node.left.value
                                    not in state_attribute_values[right_attr]
                                ):
                                    state_attribute_values[right_attr].append(
                                        node.left.value
                                    )

                        self.generic_visit(node)

                StateAttributeVisitor().visit(class_ast)
            except Exception as e:
                if verbose:
                    _printer.print(
                        f"Could not analyze class context for {function.__name__}: {e}",
                        color="yellow",
                    )
    except Exception as e:
        if verbose:
            _printer.print(
                f"Could not introspect class for {function.__name__}: {e}",
                color="yellow",
            )

    VariableAssignmentVisitor().visit(code_ast)
    ReturnVisitor().visit(code_ast)

    return list(return_values) if return_values else None


def calculate_node_levels(flow: Any) -> dict[str, int]:
    """
    Calculate the hierarchical level of each node in the flow.

    Performs a breadth-first traversal of the flow graph to assign levels
    to nodes, starting with start methods at level 0.

    Parameters
    ----------
    flow : Any
        The flow instance containing methods, listeners, and router configurations.

    Returns
    -------
    Dict[str, int]
        Dictionary mapping method names to their hierarchical levels.

    Notes
    -----
    - Start methods are assigned level 0
    - Each subsequent connected node is assigned level = parent_level + 1
    - Handles both OR and AND conditions for listeners
    - Processes router paths separately
    """
    levels: dict[str, int] = {}
    queue: deque[str] = deque()
    visited: set[str] = set()
    pending_and_listeners: dict[str, set[str]] = {}

    # Make all start methods at level 0
    for method_name, method in flow._methods.items():
        if hasattr(method, "__is_start_method__"):
            levels[method_name] = 0
            queue.append(method_name)

    # Precompute listener dependencies
    or_listeners = defaultdict(list)
    and_listeners = defaultdict(set)
    for listener_name, condition_data in flow._listeners.items():
        if isinstance(condition_data, tuple):
            condition_type, trigger_methods = condition_data
        elif isinstance(condition_data, dict):
            trigger_methods = _extract_all_methods_recursive(condition_data, flow)
            condition_type = condition_data.get("type", "OR")
        else:
            continue

        if condition_type == "OR":
            for method in trigger_methods:
                or_listeners[method].append(listener_name)
        elif condition_type == "AND":
            and_listeners[listener_name] = set(trigger_methods)

    # Breadth-first traversal to assign levels
    while queue:
        current = queue.popleft()
        current_level = levels[current]
        visited.add(current)

        for listener_name in or_listeners[current]:
            if listener_name not in levels or levels[listener_name] > current_level + 1:
                levels[listener_name] = current_level + 1
                if listener_name not in visited:
                    queue.append(listener_name)

        for listener_name, required_methods in and_listeners.items():
            if current in required_methods:
                if listener_name not in pending_and_listeners:
                    pending_and_listeners[listener_name] = set()
                pending_and_listeners[listener_name].add(current)

                if required_methods == pending_and_listeners[listener_name]:
                    if (
                        listener_name not in levels
                        or levels[listener_name] > current_level + 1
                    ):
                        levels[listener_name] = current_level + 1
                        if listener_name not in visited:
                            queue.append(listener_name)

        process_router_paths(flow, current, current_level, levels, queue)

    max_level = max(levels.values()) if levels else 0
    for method_name in flow._methods:
        if method_name not in levels:
            levels[method_name] = max_level + 1

    return levels


def count_outgoing_edges(flow: Any) -> dict[str, int]:
    """
    Count the number of outgoing edges for each method in the flow.

    Parameters
    ----------
    flow : Any
        The flow instance to analyze.

    Returns
    -------
    Dict[str, int]
        Dictionary mapping method names to their outgoing edge count.
    """
    counts = {}
    for method_name in flow._methods:
        counts[method_name] = 0
    for condition_data in flow._listeners.values():
        if isinstance(condition_data, tuple):
            _, trigger_methods = condition_data
        elif isinstance(condition_data, dict):
            trigger_methods = _extract_all_methods_recursive(condition_data, flow)
        else:
            continue

        for trigger in trigger_methods:
            if trigger in flow._methods:
                counts[trigger] += 1
    return counts


def build_ancestor_dict(flow: Any) -> dict[str, set[str]]:
    """
    Build a dictionary mapping each node to its ancestor nodes.

    Parameters
    ----------
    flow : Any
        The flow instance to analyze.

    Returns
    -------
    Dict[str, Set[str]]
        Dictionary mapping each node to a set of its ancestor nodes.
    """
    ancestors: dict[str, set[str]] = {node: set() for node in flow._methods}
    visited: set[str] = set()
    for node in flow._methods:
        if node not in visited:
            dfs_ancestors(node, ancestors, visited, flow)
    return ancestors


def dfs_ancestors(
    node: str, ancestors: dict[str, set[str]], visited: set[str], flow: Any
) -> None:
    """
    Perform depth-first search to build ancestor relationships.

    Parameters
    ----------
    node : str
        Current node being processed.
    ancestors : Dict[str, Set[str]]
        Dictionary tracking ancestor relationships.
    visited : Set[str]
        Set of already visited nodes.
    flow : Any
        The flow instance being analyzed.

    Notes
    -----
    This function modifies the ancestors dictionary in-place to build
    the complete ancestor graph.
    """
    if node in visited:
        return
    visited.add(node)

    for listener_name, condition_data in flow._listeners.items():
        if isinstance(condition_data, tuple):
            _, trigger_methods = condition_data
        elif isinstance(condition_data, dict):
            trigger_methods = _extract_all_methods_recursive(condition_data, flow)
        else:
            continue

        if node in trigger_methods:
            ancestors[listener_name].add(node)
            ancestors[listener_name].update(ancestors[node])
            dfs_ancestors(listener_name, ancestors, visited, flow)

    if node in flow._routers:
        router_method_name = node
        paths = flow._router_paths.get(router_method_name, [])
        for path in paths:
            for listener_name, condition_data in flow._listeners.items():
                if isinstance(condition_data, tuple):
                    _, trigger_methods = condition_data
                elif isinstance(condition_data, dict):
                    trigger_methods = _extract_all_methods_recursive(
                        condition_data, flow
                    )
                else:
                    continue

                if path in trigger_methods:
                    ancestors[listener_name].update(ancestors[node])
                    dfs_ancestors(listener_name, ancestors, visited, flow)


def is_ancestor(
    node: str, ancestor_candidate: str, ancestors: dict[str, set[str]]
) -> bool:
    """
    Check if one node is an ancestor of another.

    Parameters
    ----------
    node : str
        The node to check ancestors for.
    ancestor_candidate : str
        The potential ancestor node.
    ancestors : Dict[str, Set[str]]
        Dictionary containing ancestor relationships.

    Returns
    -------
    bool
        True if ancestor_candidate is an ancestor of node, False otherwise.
    """
    return ancestor_candidate in ancestors.get(node, set())


def build_parent_children_dict(flow: Any) -> dict[str, list[str]]:
    """
    Build a dictionary mapping parent nodes to their children.

    Parameters
    ----------
    flow : Any
        The flow instance to analyze.

    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping parent method names to lists of their child method names.

    Notes
    -----
    - Maps listeners to their trigger methods
    - Maps router methods to their paths and listeners
    - Children lists are sorted for consistent ordering
    """
    parent_children: dict[str, list[str]] = {}

    for listener_name, condition_data in flow._listeners.items():
        if isinstance(condition_data, tuple):
            _, trigger_methods = condition_data
        elif isinstance(condition_data, dict):
            trigger_methods = _extract_all_methods_recursive(condition_data, flow)
        else:
            continue

        for trigger in trigger_methods:
            if trigger not in parent_children:
                parent_children[trigger] = []
            if listener_name not in parent_children[trigger]:
                parent_children[trigger].append(listener_name)

    for router_method_name, paths in flow._router_paths.items():
        for path in paths:
            for listener_name, condition_data in flow._listeners.items():
                if isinstance(condition_data, tuple):
                    _, trigger_methods = condition_data
                elif isinstance(condition_data, dict):
                    trigger_methods = _extract_all_methods_recursive(
                        condition_data, flow
                    )
                else:
                    continue

                if path in trigger_methods:
                    if router_method_name not in parent_children:
                        parent_children[router_method_name] = []
                    if listener_name not in parent_children[router_method_name]:
                        parent_children[router_method_name].append(listener_name)

    return parent_children


def get_child_index(
    parent: str, child: str, parent_children: dict[str, list[str]]
) -> int:
    """
    Get the index of a child node in its parent's sorted children list.

    Parameters
    ----------
    parent : str
        The parent node name.
    child : str
        The child node name to find the index for.
    parent_children : Dict[str, List[str]]
        Dictionary mapping parents to their children lists.

    Returns
    -------
    int
        Zero-based index of the child in its parent's sorted children list.
    """
    children = parent_children.get(parent, [])
    children.sort()
    return children.index(child)


def process_router_paths(
    flow: Any,
    current: str,
    current_level: int,
    levels: dict[str, int],
    queue: deque[str],
) -> None:
    """Handle the router connections for the current node."""
    if current in flow._routers:
        paths = flow._router_paths.get(current, [])
        for path in paths:
            for listener_name, condition_data in flow._listeners.items():
                if isinstance(condition_data, tuple):
                    _condition_type, trigger_methods = condition_data
                elif isinstance(condition_data, dict):
                    trigger_methods = _extract_all_methods_recursive(
                        condition_data, flow
                    )
                else:
                    continue

                if path in trigger_methods:
                    if (
                        listener_name not in levels
                        or levels[listener_name] > current_level + 1
                    ):
                        levels[listener_name] = current_level + 1
                        queue.append(listener_name)


def is_flow_method_name(obj: Any) -> TypeIs[FlowMethodName]:
    """Check if the object is a valid flow method name.

    Args:
        obj: The object to check.
    Returns:
        True if the object is a valid flow method name, False otherwise.
    """
    return isinstance(obj, str)


def is_flow_method_callable(obj: Any) -> TypeIs[FlowMethodCallable[..., Any]]:
    """Check if the object is a callable flow method.

    Args:
        obj: The object to check.

    Returns:
        True if the object is a callable, False otherwise.
    """
    return callable(obj) and hasattr(obj, "__name__")


def is_flow_condition_list(obj: Any) -> TypeIs[FlowConditions]:
    """Check if the object is a list of FlowCondition dictionaries.

    Args:
        obj: The object to check.

    Returns:
        True if the object is a list of FlowCondition dictionaries, False otherwise.
    """
    if not isinstance(obj, list):
        return False

    for item in obj:
        if not (is_flow_method_name(item) or is_flow_condition_dict(item)):
            return False

    return True


def is_simple_flow_condition(obj: Any) -> TypeIs[SimpleFlowCondition]:
    """Check if the object is a simple flow condition tuple.

    Args:
        obj: The object to check.

    Returns:
        True if the object is a (condition_type, methods) tuple, False otherwise.
    """
    return (
        isinstance(obj, tuple)
        and len(obj) == 2
        and isinstance(obj[0], str)
        and isinstance(obj[1], list)
    )


def is_flow_method(obj: Any) -> TypeIs[FlowMethod[Any, Any]]:
    """Check if the object is a flow method wrapper.

    Checks for attributes added by @start, @listen, or @router decorators.

    Args:
        obj: The object to check.

    Returns:
        True if the object is a FlowMethod subclass (StartMethod, ListenMethod, or RouterMethod).
    """
    return (
        hasattr(obj, "__is_flow_method__")
        or hasattr(obj, "__is_start_method__")
        or hasattr(obj, "__trigger_methods__")
        or hasattr(obj, "__is_router__")
    )


def is_flow_condition_dict(obj: Any) -> TypeIs[FlowCondition]:
    """Check if the object matches the FlowCondition structure.

    Args:
        obj: The object to check.

    Returns:
        True if the object is a valid FlowCondition dictionary, False otherwise.
    """
    if not isinstance(obj, dict):
        return False

    type_value = obj.get("type")
    if type_value not in ("AND", "OR"):
        return False

    if "conditions" in obj:
        conditions = obj["conditions"]
        if not isinstance(conditions, list):
            return False
        for cond in conditions:
            if not (
                isinstance(cond, str)
                or (isinstance(cond, dict) and is_flow_condition_dict(cond))
            ):
                return False

    if "methods" in obj:
        methods = obj["methods"]
        if not (isinstance(methods, list) and all(isinstance(m, str) for m in methods)):
            return False

    allowed_keys = {"type", "conditions", "methods"}
    if not set(obj).issubset(allowed_keys):
        return False

    return True


def _extract_all_methods_recursive(
    condition: str | FlowCondition | dict[str, Any] | list[Any],
    flow: Flow[Any] | None = None,
) -> list[FlowMethodName]:
    """Extract ALL method names from a condition tree recursively.

    This function recursively extracts every method name from the entire
    condition tree, regardless of nesting. Used for visualization and debugging.

    Note: Only extracts actual method names, not router output strings.
    If flow is provided, it will filter out strings that are not in flow._methods.

    Args:
        condition: Can be a string, dict, or list
        flow: Optional flow instance to filter out non-method strings

    Returns:
        List of all method names found in the condition tree
    """
    if is_flow_method_name(condition):
        if flow is not None:
            if condition in flow._methods:
                return [condition]
            return []
        return [condition]
    if is_flow_condition_dict(condition):
        normalized = _normalize_condition(condition)
        methods = []
        for sub_cond in normalized.get("conditions", []):
            methods.extend(_extract_all_methods_recursive(sub_cond, flow))
        return methods
    if isinstance(condition, list):
        methods = []
        for item in condition:
            methods.extend(_extract_all_methods_recursive(item, flow))
        return methods
    return []


def _normalize_condition(
    condition: FlowConditions | FlowCondition | FlowMethodName,
) -> FlowCondition:
    """Normalize a condition to standard format with 'conditions' key.

    Args:
        condition: Can be a string (method name), dict (condition), or list

    Returns:
        Normalized dict with 'type' and 'conditions' keys
    """
    if is_flow_method_name(condition):
        return {"type": OR_CONDITION, "conditions": [condition]}
    if is_flow_condition_dict(condition):
        if "conditions" in condition:
            return condition
        if "methods" in condition:
            return {"type": condition["type"], "conditions": condition["methods"]}
        return condition
    if is_flow_condition_list(condition):
        return {"type": OR_CONDITION, "conditions": condition}

    raise ValueError(f"Cannot normalize condition: {condition}")


def _extract_all_methods(
    condition: str | FlowCondition | dict[str, Any] | list[Any],
) -> list[FlowMethodName]:
    """Extract all method names from a condition (including nested).

    For AND conditions, this extracts methods that must ALL complete.
    For OR conditions nested inside AND, we don't extract their methods
    since only one branch of the OR needs to trigger, not all methods.

    This function is used for runtime execution logic, where we need to know
    which methods must complete for AND conditions. For visualization purposes,
    use _extract_all_methods_recursive() instead.

    Args:
        condition: Can be a string, dict, or list

    Returns:
        List of all method names in the condition tree that must complete
    """
    if is_flow_method_name(condition):
        return [condition]
    if is_flow_condition_dict(condition):
        normalized = _normalize_condition(condition)
        cond_type = normalized.get("type", OR_CONDITION)

        if cond_type == AND_CONDITION:
            return [
                sub_cond
                for sub_cond in normalized.get("conditions", [])
                if is_flow_method_name(sub_cond)
            ]
        return []
    if isinstance(condition, list):
        methods = []
        for item in condition:
            methods.extend(_extract_all_methods(item))
        return methods
    return []
