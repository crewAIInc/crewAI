import asyncio
import inspect
from typing import Any, Callable, Dict, Generic, List, Set, Type, TypeVar, Union

import graphviz
from pydantic import BaseModel

T = TypeVar("T", bound=Union[BaseModel, Dict[str, Any]])


def start(condition=None):
    def decorator(func):
        func.__is_start_method__ = True
        if condition is not None:
            if isinstance(condition, str):
                func.__trigger_methods__ = [condition]
                func.__condition_type__ = "OR"
            elif (
                isinstance(condition, dict)
                and "type" in condition
                and "methods" in condition
            ):
                func.__trigger_methods__ = condition["methods"]
                func.__condition_type__ = condition["type"]
            elif callable(condition) and hasattr(condition, "__name__"):
                func.__trigger_methods__ = [condition.__name__]
                func.__condition_type__ = "OR"
            else:
                raise ValueError(
                    "Condition must be a method, string, or a result of or_() or and_()"
                )
        return func

    return decorator


def listen(condition):
    def decorator(func):
        if isinstance(condition, str):
            func.__trigger_methods__ = [condition]
            func.__condition_type__ = "OR"
        elif (
            isinstance(condition, dict)
            and "type" in condition
            and "methods" in condition
        ):
            func.__trigger_methods__ = condition["methods"]
            func.__condition_type__ = condition["type"]
        elif callable(condition) and hasattr(condition, "__name__"):
            func.__trigger_methods__ = [condition.__name__]
            func.__condition_type__ = "OR"
        else:
            raise ValueError(
                "Condition must be a method, string, or a result of or_() or and_()"
            )
        return func

    return decorator


def router(method):
    def decorator(func):
        func.__is_router__ = True
        func.__router_for__ = method.__name__
        return func

    return decorator


def or_(*conditions):
    methods = []
    for condition in conditions:
        if isinstance(condition, dict) and "methods" in condition:
            methods.extend(condition["methods"])
        elif isinstance(condition, str):
            methods.append(condition)
        elif callable(condition):
            methods.append(getattr(condition, "__name__", repr(condition)))
        else:
            raise ValueError("Invalid condition in or_()")
    return {"type": "OR", "methods": methods}


def and_(*conditions):
    methods = []
    for condition in conditions:
        if isinstance(condition, dict) and "methods" in condition:
            methods.extend(condition["methods"])
        elif isinstance(condition, str):
            methods.append(condition)
        elif callable(condition):
            methods.append(getattr(condition, "__name__", repr(condition)))
        else:
            raise ValueError("Invalid condition in and_()")
    return {"type": "AND", "methods": methods}


class FlowMeta(type):
    def __new__(mcs, name, bases, dct):
        cls = super().__new__(mcs, name, bases, dct)

        start_methods = []
        listeners = {}
        routers = {}

        for attr_name, attr_value in dct.items():
            if hasattr(attr_value, "__is_start_method__"):
                start_methods.append(attr_name)
                if hasattr(attr_value, "__trigger_methods__"):
                    methods = attr_value.__trigger_methods__
                    condition_type = getattr(attr_value, "__condition_type__", "OR")
                    listeners[attr_name] = (condition_type, methods)
            elif hasattr(attr_value, "__trigger_methods__"):
                methods = attr_value.__trigger_methods__
                condition_type = getattr(attr_value, "__condition_type__", "OR")
                listeners[attr_name] = (condition_type, methods)
            elif hasattr(attr_value, "__is_router__"):
                routers[attr_value.__router_for__] = attr_name

        setattr(cls, "_start_methods", start_methods)
        setattr(cls, "_listeners", listeners)
        setattr(cls, "_routers", routers)

        return cls


class Flow(Generic[T], metaclass=FlowMeta):
    _start_methods: List[str] = []
    _listeners: Dict[str, tuple[str, List[str]]] = {}
    _routers: Dict[str, str] = {}
    initial_state: Union[Type[T], T, None] = None

    def __class_getitem__(cls, item):
        class _FlowGeneric(cls):
            _initial_state_T = item

        return _FlowGeneric

    def __init__(self):
        self._methods: Dict[str, Callable] = {}
        self._state = self._create_initial_state()
        self._completed_methods: Set[str] = set()
        self._pending_and_listeners: Dict[str, Set[str]] = {}
        self._method_outputs: List[Any] = []  # List to store all method outputs

        for method_name in dir(self):
            if callable(getattr(self, method_name)) and not method_name.startswith(
                "__"
            ):
                self._methods[method_name] = getattr(self, method_name)

    def _create_initial_state(self) -> T:
        if self.initial_state is None and hasattr(self, "_initial_state_T"):
            return self._initial_state_T()  # type: ignore
        if self.initial_state is None:
            return {}  # type: ignore
        elif isinstance(self.initial_state, type):
            return self.initial_state()
        else:
            return self.initial_state

    @property
    def state(self) -> T:
        return self._state

    @property
    def method_outputs(self) -> List[Any]:
        """Returns the list of all outputs from executed methods."""
        return self._method_outputs

    async def kickoff(self) -> Any:
        if not self._start_methods:
            raise ValueError("No start method defined")

        # Create tasks for all start methods
        tasks = [
            self._execute_start_method(start_method)
            for start_method in self._start_methods
        ]

        # Run all start methods concurrently
        await asyncio.gather(*tasks)

        # Return the final output (from the last executed method)
        if self._method_outputs:
            return self._method_outputs[-1]
        else:
            return None  # Or raise an exception if no methods were executed

    async def _execute_start_method(self, start_method: str):
        result = await self._execute_method(self._methods[start_method])
        await self._execute_listeners(start_method, result)

    async def _execute_method(self, method: Callable, *args, **kwargs):
        result = (
            await method(*args, **kwargs)
            if asyncio.iscoroutinefunction(method)
            else method(*args, **kwargs)
        )
        self._method_outputs.append(result)  # Store the output
        return result

    async def _execute_listeners(self, trigger_method: str, result: Any):
        listener_tasks = []

        if trigger_method in self._routers:
            router_method = self._methods[self._routers[trigger_method]]
            path = await self._execute_method(router_method)
            # Use the path as the new trigger method
            trigger_method = path

        for listener, (condition_type, methods) in self._listeners.items():
            if condition_type == "OR":
                if trigger_method in methods:
                    listener_tasks.append(
                        self._execute_single_listener(listener, result)
                    )
            elif condition_type == "AND":
                if listener not in self._pending_and_listeners:
                    self._pending_and_listeners[listener] = set()
                self._pending_and_listeners[listener].add(trigger_method)
                if set(methods) == self._pending_and_listeners[listener]:
                    listener_tasks.append(
                        self._execute_single_listener(listener, result)
                    )
                    del self._pending_and_listeners[listener]

        # Run all listener tasks concurrently and wait for them to complete
        await asyncio.gather(*listener_tasks)

    async def _execute_single_listener(self, listener: str, result: Any):
        try:
            method = self._methods[listener]
            sig = inspect.signature(method)
            params = list(sig.parameters.values())

            # Exclude 'self' parameter
            method_params = [p for p in params if p.name != "self"]

            if method_params:
                # If listener expects parameters, pass the result
                listener_result = await self._execute_method(method, result)
            else:
                # If listener does not expect parameters, call without arguments
                listener_result = await self._execute_method(method)

            # Execute listeners of this listener
            await self._execute_listeners(listener, listener_result)
        except Exception as e:
            print(f"[Flow._execute_single_listener] Error in method {listener}: {e}")
            import traceback

            traceback.print_exc()

    def visualize(self, filename="flow_graph"):
        dot = graphviz.Digraph(comment="Flow Graph", engine="dot")
        dot.attr(
            rankdir="TB", size="20,20", splines="curved"
        )  # Changed to curved splines

        # Color scheme (using company colors)
        colors = {
            "bg": "#FFFFFF",
            "start": "#FF5A50",
            "method": "#333333",
            "router_outline": "#FF5A50",
            "edge": "#333333",
            "text": "#FFFFFF",
        }

        dot.attr(bgcolor=colors["bg"])

        # Add nodes for each relevant method
        for method_name, method in self._methods.items():
            if (
                hasattr(method, "__is_start_method__")
                or method_name in self._listeners
                or method_name in self._routers.values()
            ):
                shape = "rectangle"
                style = "filled,rounded"
                fillcolor = colors["method"]

                if hasattr(method, "__is_start_method__"):
                    fillcolor = colors["start"]

                dot.node(
                    method_name,
                    method_name,
                    shape=shape,
                    style=style,
                    fillcolor=fillcolor,
                    fontcolor=colors["text"],
                    penwidth="2",
                )

        # Add edges and enhanced router representation
        for method_name, method in self._methods.items():
            if method_name in self._listeners:
                condition_type, trigger_methods = self._listeners[method_name]
                for trigger in trigger_methods:
                    if condition_type == "AND":
                        dot.edge(
                            trigger,
                            method_name,
                            color=colors["edge"],
                            style="dashed",
                            penwidth="2",
                        )
                    else:  # OR condition
                        dot.edge(
                            trigger, method_name, color=colors["edge"], penwidth="2"
                        )

            if method_name in self._routers.values():
                for trigger, router in self._routers.items():
                    if router == method_name:
                        # Create a subgraph for the router and its outputs
                        subgraph_name = f"cluster_{method_name}"
                        subgraph = graphviz.Digraph(name=subgraph_name)
                        subgraph.attr(
                            label="",
                            style="filled,rounded",
                            color=colors["router_outline"],
                            fillcolor=colors["method"],
                            penwidth="3",
                        )  # Changed to solid line and increased penwidth

                        # Router label (method name) and outputs
                        label = f"{method_name}\\n\\nPossible outcomes:\\n• Success\\n• Failure"
                        subgraph.node(
                            method_name,
                            label,
                            shape="plaintext",
                            fontcolor=colors["text"],
                        )

                        # Add the subgraph to the main graph
                        dot.subgraph(subgraph)

                        # Connect trigger to router (to the border of the subgraph)
                        dot.edge(
                            trigger,
                            method_name,
                            color=colors["edge"],
                            style="solid",
                            penwidth="2",
                            lhead=subgraph_name,
                        )

        # Render and save the graph
        dot.render(filename, format="png", cleanup=True, view=True)
        print(f"Graph saved as {filename}.png")
