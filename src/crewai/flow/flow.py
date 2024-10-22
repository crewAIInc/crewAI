# flow.py

import asyncio
import inspect
from typing import Any, Callable, Dict, Generic, List, Set, Type, TypeVar, Union

from pydantic import BaseModel

from crewai.flow.flow_visualizer import plot_flow
from crewai.flow.utils import get_possible_return_constants
from crewai.telemetry import Telemetry

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
        router_paths = {}

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
                possible_returns = get_possible_return_constants(attr_value)
                if possible_returns:
                    router_paths[attr_name] = possible_returns

                # Register router as a listener to its triggering method
                trigger_method_name = attr_value.__router_for__
                methods = [trigger_method_name]
                condition_type = "OR"
                listeners[attr_name] = (condition_type, methods)

        setattr(cls, "_start_methods", start_methods)
        setattr(cls, "_listeners", listeners)
        setattr(cls, "_routers", routers)
        setattr(cls, "_router_paths", router_paths)

        return cls


class Flow(Generic[T], metaclass=FlowMeta):
    _telemetry = Telemetry()

    _start_methods: List[str] = []
    _listeners: Dict[str, tuple[str, List[str]]] = {}
    _routers: Dict[str, str] = {}
    _router_paths: Dict[str, List[str]] = {}
    initial_state: Union[Type[T], T, None] = None

    def __class_getitem__(cls: Type["Flow"], item: Type[T]) -> Type["Flow"]:
        class _FlowGeneric(cls):  # type: ignore
            _initial_state_T = item  # type: ignore

        _FlowGeneric.__name__ = f"{cls.__name__}[{item.__name__}]"
        return _FlowGeneric

    def __init__(self) -> None:
        self._methods: Dict[str, Callable] = {}
        self._state: T = self._create_initial_state()
        self._completed_methods: Set[str] = set()
        self._pending_and_listeners: Dict[str, Set[str]] = {}
        self._method_outputs: List[Any] = []  # List to store all method outputs

        self._telemetry.flow_creation_span(self.__class__.__name__)

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

    def kickoff(self) -> Any:
        return asyncio.run(self.kickoff_async())

    async def kickoff_async(self) -> Any:
        if not self._start_methods:
            raise ValueError("No start method defined")

        self._telemetry.flow_execution_span(
            self.__class__.__name__, list(self._methods.keys())
        )

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

    async def _execute_start_method(self, start_method: str) -> None:
        result = await self._execute_method(self._methods[start_method])
        await self._execute_listeners(start_method, result)

    async def _execute_method(self, method: Callable, *args: Any, **kwargs: Any) -> Any:
        result = (
            await method(*args, **kwargs)
            if asyncio.iscoroutinefunction(method)
            else method(*args, **kwargs)
        )
        self._method_outputs.append(result)  # Store the output
        return result

    async def _execute_listeners(self, trigger_method: str, result: Any) -> None:
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

    async def _execute_single_listener(self, listener: str, result: Any) -> None:
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

    def plot(self, filename: str = "crewai_flow") -> None:
        self._telemetry.flow_plotting_span(
            self.__class__.__name__, list(self._methods.keys())
        )

        plot_flow(self, filename)
