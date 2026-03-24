"""Flow structure serializer for introspecting Flow classes.

This module provides the flow_structure() function that analyzes a Flow class
and returns a JSON-serializable dictionary describing its graph structure.
This is used by Studio UI to render a visual flow graph.

Example:
    >>> from crewai.flow import Flow, start, listen
    >>> from crewai.flow.flow_serializer import flow_structure
    >>>
    >>> class MyFlow(Flow):
    ...     @start()
    ...     def begin(self):
    ...         return "started"
    ...
    ...     @listen(begin)
    ...     def process(self):
    ...         return "done"
    >>>
    >>> structure = flow_structure(MyFlow)
    >>> print(structure["name"])
    'MyFlow'
"""

from __future__ import annotations

import inspect
import logging
import re
import textwrap
from typing import Any, TypedDict, get_args, get_origin

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from crewai.flow.flow_wrappers import (
    FlowCondition,
    FlowMethod,
    ListenMethod,
    RouterMethod,
    StartMethod,
)


logger = logging.getLogger(__name__)


class MethodInfo(TypedDict, total=False):
    """Information about a single flow method.

    Attributes:
        name: The method name.
        type: Method type - start, listen, router, or start_router.
        trigger_methods: List of method names that trigger this method.
        condition_type: 'AND' or 'OR' for composite conditions, null otherwise.
        router_paths: For routers, the possible route names returned.
        has_human_feedback: Whether the method has @human_feedback decorator.
        has_crew: Whether the method body references a Crew.
    """

    name: str
    type: str
    trigger_methods: list[str]
    condition_type: str | None
    router_paths: list[str]
    has_human_feedback: bool
    has_crew: bool


class EdgeInfo(TypedDict, total=False):
    """Information about an edge between flow methods.

    Attributes:
        from_method: Source method name.
        to_method: Target method name.
        edge_type: Type of edge - 'listen' or 'route'.
        condition: Route name for router edges, null for listen edges.
    """

    from_method: str
    to_method: str
    edge_type: str
    condition: str | None


class StateFieldInfo(TypedDict, total=False):
    """Information about a state field.

    Attributes:
        name: Field name.
        type: Field type as string.
        default: Default value if any.
    """

    name: str
    type: str
    default: Any


class StateSchemaInfo(TypedDict, total=False):
    """Information about the flow's state schema.

    Attributes:
        fields: List of field information.
    """

    fields: list[StateFieldInfo]


class FlowStructureInfo(TypedDict, total=False):
    """Complete flow structure information.

    Attributes:
        name: Flow class name.
        description: Flow docstring if available.
        methods: List of method information.
        edges: List of edge information.
        state_schema: State schema if typed, null otherwise.
        inputs: Detected flow inputs if available.
    """

    name: str
    description: str | None
    methods: list[MethodInfo]
    edges: list[EdgeInfo]
    state_schema: StateSchemaInfo | None
    inputs: list[str]


def _get_method_type(
    method_name: str,
    method: Any,
    start_methods: list[str],
    routers: set[str],
) -> str:
    """Determine the type of a flow method.

    Args:
        method_name: Name of the method.
        method: The method object.
        start_methods: List of start method names.
        routers: Set of router method names.

    Returns:
        One of: 'start', 'listen', 'router', or 'start_router'.
    """
    is_start = method_name in start_methods or getattr(
        method, "__is_start_method__", False
    )
    is_router = method_name in routers or getattr(method, "__is_router__", False)

    if is_start and is_router:
        return "start_router"
    if is_start:
        return "start"
    if is_router:
        return "router"
    return "listen"


def _has_human_feedback(method: Any) -> bool:
    """Check if a method has the @human_feedback decorator.

    Args:
        method: The method object to check.

    Returns:
        True if the method has __human_feedback_config__ attribute.
    """
    return hasattr(method, "__human_feedback_config__")


def _detect_crew_reference(method: Any) -> bool:
    """Detect if a method body references a Crew.

    Checks for patterns like:
    - .crew() method calls
    - Crew( instantiation
    - References to Crew class in type hints

    Note:
        This is a **best-effort heuristic for UI hints**, not a guarantee.
        Uses inspect.getsource + regex which can false-positive on comments
        or string literals, and may fail on dynamically generated methods
        or lambdas. Do not rely on this for correctness-critical logic.

    Args:
        method: The method object to inspect.

    Returns:
        True if crew reference detected, False otherwise.
    """
    try:
        # Get the underlying function from wrapper
        func = method
        if hasattr(method, "_meth"):
            func = method._meth
        elif hasattr(method, "__wrapped__"):
            func = method.__wrapped__

        source = inspect.getsource(func)
        source = textwrap.dedent(source)

        # Patterns that indicate Crew usage
        crew_patterns = [
            r"\.crew\(\)",  # .crew() method call
            r"Crew\s*\(",  # Crew( instantiation
            r":\s*Crew\b",  # Type hint with Crew
            r"->.*Crew",  # Return type hint with Crew
        ]

        for pattern in crew_patterns:
            if re.search(pattern, source):
                return True

        return False
    except (OSError, TypeError):
        # Can't get source code - assume no crew reference
        return False


def _extract_trigger_methods(method: Any) -> tuple[list[str], str | None]:
    """Extract trigger methods and condition type from a method.

    Args:
        method: The method object to inspect.

    Returns:
        Tuple of (trigger_methods list, condition_type or None).
    """
    trigger_methods: list[str] = []
    condition_type: str | None = None

    # First try __trigger_methods__ (populated for simple conditions)
    if hasattr(method, "__trigger_methods__") and method.__trigger_methods__:
        trigger_methods = [str(m) for m in method.__trigger_methods__]

    # For complex conditions (or_/and_ combinators), extract from __trigger_condition__
    if (
        not trigger_methods
        and hasattr(method, "__trigger_condition__")
        and method.__trigger_condition__
    ):
        trigger_condition = method.__trigger_condition__
        trigger_methods = _extract_all_methods_from_condition(trigger_condition)

    if hasattr(method, "__condition_type__") and method.__condition_type__:
        condition_type = str(method.__condition_type__)

    return trigger_methods, condition_type


def _extract_router_paths(
    method: Any, router_paths_registry: dict[str, list[str]]
) -> list[str]:
    """Extract router paths for a router method.

    Args:
        method: The method object.
        router_paths_registry: The class-level _router_paths dict.

    Returns:
        List of possible route names.
    """
    method_name = getattr(method, "__name__", "")

    # First check if there are __router_paths__ on the method itself
    if hasattr(method, "__router_paths__") and method.__router_paths__:
        return [str(p) for p in method.__router_paths__]

    # Then check the class-level registry
    if method_name in router_paths_registry:
        return [str(p) for p in router_paths_registry[method_name]]

    return []


def _extract_all_methods_from_condition(
    condition: str | FlowCondition | dict[str, Any] | list[Any],
) -> list[str]:
    """Extract all method names from a condition tree recursively.

    Args:
        condition: Can be a string, FlowCondition tuple, dict, or list.

    Returns:
        List of all method names found in the condition.
    """
    if isinstance(condition, str):
        return [condition]
    if isinstance(condition, tuple) and len(condition) == 2:
        # FlowCondition: (condition_type, methods_list)
        _, methods = condition
        if isinstance(methods, list):
            result: list[str] = []
            for m in methods:
                result.extend(_extract_all_methods_from_condition(m))
            return result
        return []
    if isinstance(condition, dict):
        conditions_list = condition.get("conditions", [])
        dict_methods: list[str] = []
        for sub_cond in conditions_list:
            dict_methods.extend(_extract_all_methods_from_condition(sub_cond))
        return dict_methods
    if isinstance(condition, list):
        list_methods: list[str] = []
        for item in condition:
            list_methods.extend(_extract_all_methods_from_condition(item))
        return list_methods
    return []


def _generate_edges(
    listeners: dict[str, tuple[str, list[str]] | FlowCondition],
    routers: set[str],
    router_paths: dict[str, list[str]],
    all_methods: set[str],
) -> list[EdgeInfo]:
    """Generate edges from listeners and routers.

    Args:
        listeners: Map of listener_name -> (condition_type, trigger_methods) or FlowCondition.
        routers: Set of router method names.
        router_paths: Map of router_name -> possible return values.
        all_methods: Set of all method names in the flow.

    Returns:
        List of EdgeInfo dictionaries.
    """
    edges: list[EdgeInfo] = []

    # Generate edges from listeners (listen edges)
    for listener_name, condition_data in listeners.items():
        trigger_methods: list[str] = []

        if isinstance(condition_data, tuple) and len(condition_data) == 2:
            _condition_type, methods = condition_data
            trigger_methods = [str(m) for m in methods]
        elif isinstance(condition_data, dict):
            trigger_methods = _extract_all_methods_from_condition(condition_data)

        # Create edges from each trigger to the listener
        edges.extend(
            EdgeInfo(
                from_method=trigger,
                to_method=listener_name,
                edge_type="listen",
                condition=None,
            )
            for trigger in trigger_methods
            if trigger in all_methods
        )

    # Generate edges from routers (route edges)
    for router_name, paths in router_paths.items():
        for path in paths:
            # Find listeners that listen to this path
            for listener_name, condition_data in listeners.items():
                path_triggers: list[str] = []

                if isinstance(condition_data, tuple) and len(condition_data) == 2:
                    _, methods = condition_data
                    path_triggers = [str(m) for m in methods]
                elif isinstance(condition_data, dict):
                    path_triggers = _extract_all_methods_from_condition(condition_data)

                if str(path) in path_triggers:
                    edges.append(
                        EdgeInfo(
                            from_method=router_name,
                            to_method=listener_name,
                            edge_type="route",
                            condition=str(path),
                        )
                    )

    return edges


def _extract_state_schema(flow_class: type) -> StateSchemaInfo | None:
    """Extract state schema from a Flow class.

    Checks for:
    - Generic type parameter (Flow[MyState])
    - initial_state class attribute

    Args:
        flow_class: The Flow class to inspect.

    Returns:
        StateSchemaInfo if a Pydantic model state is detected, None otherwise.
    """
    state_type: type | None = None

    # Check for _initial_state_t set by __class_getitem__
    if hasattr(flow_class, "_initial_state_t"):
        state_type = flow_class._initial_state_t

    # Check initial_state class attribute
    if state_type is None and hasattr(flow_class, "initial_state"):
        initial_state = flow_class.initial_state
        if isinstance(initial_state, type) and issubclass(initial_state, BaseModel):
            state_type = initial_state
        elif isinstance(initial_state, BaseModel):
            state_type = type(initial_state)

    # Check __orig_bases__ for generic parameters
    if state_type is None and hasattr(flow_class, "__orig_bases__"):
        for base in flow_class.__orig_bases__:
            origin = get_origin(base)
            if origin is not None:
                args = get_args(base)
                if args:
                    candidate = args[0]
                    if isinstance(candidate, type) and issubclass(candidate, BaseModel):
                        state_type = candidate
                        break

    if state_type is None or not issubclass(state_type, BaseModel):
        return None

    # Extract fields from the Pydantic model
    fields: list[StateFieldInfo] = []
    try:
        model_fields = state_type.model_fields
        for field_name, field_info in model_fields.items():
            field_type_str = "Any"
            if field_info.annotation is not None:
                field_type_str = str(field_info.annotation)
                # Clean up the type string
                field_type_str = field_type_str.replace("typing.", "")
                field_type_str = field_type_str.replace("<class '", "").replace(
                    "'>", ""
                )

            default_value = None
            if (
                field_info.default is not PydanticUndefined
                and field_info.default is not None
                and not callable(field_info.default)
            ):
                try:
                    # Try to serialize the default value
                    default_value = field_info.default
                except Exception:
                    default_value = str(field_info.default)

            fields.append(
                StateFieldInfo(
                    name=field_name,
                    type=field_type_str,
                    default=default_value,
                )
            )
    except Exception:
        logger.debug(
            "Failed to extract state schema fields for %s", flow_class.__name__
        )

    return StateSchemaInfo(fields=fields) if fields else None


def _detect_flow_inputs(flow_class: type) -> list[str]:
    """Detect flow input parameters.

    Inspects the __init__ signature for custom parameters beyond standard Flow params.

    Args:
        flow_class: The Flow class to inspect.

    Returns:
        List of detected input names.
    """
    inputs: list[str] = []

    # Check for inputs in __init__ signature beyond standard Flow params
    try:
        init_method = flow_class.__init__  # type: ignore[misc]
        init_sig = inspect.signature(init_method)
        standard_params = {
            "self",
            "persistence",
            "tracing",
            "suppress_flow_events",
            "max_method_calls",
            "kwargs",
        }
        inputs.extend(
            param_name
            for param_name in init_sig.parameters
            if param_name not in standard_params and not param_name.startswith("_")
        )
    except Exception:
        logger.debug(
            "Failed to detect inputs from __init__ for %s", flow_class.__name__
        )

    return inputs


def flow_structure(flow_class: type) -> FlowStructureInfo:
    """Introspect a Flow class and return its structure as a JSON-serializable dict.

    This function analyzes a Flow CLASS (not instance) and returns complete
    information about its graph structure including methods, edges, and state.

    Args:
        flow_class: A Flow class (not an instance) to introspect.

    Returns:
        FlowStructureInfo dictionary containing:
        - name: Flow class name
        - description: Docstring if available
        - methods: List of method info dicts
        - edges: List of edge info dicts
        - state_schema: State schema if typed, None otherwise
        - inputs: Detected input names

    Raises:
        TypeError: If flow_class is not a class.

    Example:
        >>> structure = flow_structure(MyFlow)
        >>> print(structure["name"])
        'MyFlow'
        >>> for method in structure["methods"]:
        ...     print(method["name"], method["type"])
    """
    if not isinstance(flow_class, type):
        raise TypeError(
            f"flow_structure requires a Flow class, not an instance. "
            f"Got {type(flow_class).__name__}"
        )

    # Get class-level metadata set by FlowMeta
    start_methods: list[str] = getattr(flow_class, "_start_methods", [])
    listeners: dict[str, Any] = getattr(flow_class, "_listeners", {})
    routers: set[str] = getattr(flow_class, "_routers", set())
    router_paths_registry: dict[str, list[str]] = getattr(
        flow_class, "_router_paths", {}
    )

    # Collect all flow methods
    methods: list[MethodInfo] = []
    all_method_names: set[str] = set()

    for attr_name in dir(flow_class):
        if attr_name.startswith("_"):
            continue

        try:
            attr = getattr(flow_class, attr_name)
        except AttributeError:
            continue

        # Check if it's a flow method
        is_flow_method = (
            isinstance(attr, (FlowMethod, StartMethod, ListenMethod, RouterMethod))
            or hasattr(attr, "__is_flow_method__")
            or hasattr(attr, "__is_start_method__")
            or hasattr(attr, "__trigger_methods__")
            or hasattr(attr, "__is_router__")
        )

        if not is_flow_method:
            continue

        all_method_names.add(attr_name)

        # Get method type
        method_type = _get_method_type(attr_name, attr, start_methods, routers)

        # Get trigger methods and condition type
        trigger_methods, condition_type = _extract_trigger_methods(attr)

        # Get router paths if applicable
        router_paths_list: list[str] = []
        if method_type in ("router", "start_router"):
            router_paths_list = _extract_router_paths(attr, router_paths_registry)

        # Check for human feedback
        has_hf = _has_human_feedback(attr)

        # Check for crew reference
        has_crew = _detect_crew_reference(attr)

        method_info = MethodInfo(
            name=attr_name,
            type=method_type,
            trigger_methods=trigger_methods,
            condition_type=condition_type,
            router_paths=router_paths_list,
            has_human_feedback=has_hf,
            has_crew=has_crew,
        )
        methods.append(method_info)

    # Generate edges
    edges = _generate_edges(listeners, routers, router_paths_registry, all_method_names)

    # Extract state schema
    state_schema = _extract_state_schema(flow_class)

    # Detect inputs
    inputs = _detect_flow_inputs(flow_class)

    # Get flow description from docstring
    description: str | None = None
    if flow_class.__doc__:
        description = flow_class.__doc__.strip()

    return FlowStructureInfo(
        name=flow_class.__name__,
        description=description,
        methods=methods,
        edges=edges,
        state_schema=state_schema,
        inputs=inputs,
    )
