import asyncio
import inspect
from typing import Any, Callable, Dict, Generic, List, Type, TypeVar, Union

from pydantic import BaseModel

T = TypeVar("T", bound=Union[BaseModel, Dict[str, Any]])


class FlowMeta(type):
    def __new__(mcs, name, bases, dct):
        cls = super().__new__(mcs, name, bases, dct)

        start_methods = []
        listeners = {}

        for attr_name, attr_value in dct.items():
            if hasattr(attr_value, "__is_start_method__"):
                start_methods.append(attr_name)
            if hasattr(attr_value, "__trigger_methods__"):
                condition = attr_value.__trigger_methods__
                if callable(condition):
                    # Single method reference
                    method_name = condition.__name__
                    if method_name not in listeners:
                        listeners[method_name] = []
                    listeners[method_name].append((attr_name, "SINGLE", [method_name]))
                elif isinstance(condition, str):
                    # Single method name
                    if condition not in listeners:
                        listeners[condition] = []
                    listeners[condition].append((attr_name, "SINGLE", [condition]))
                elif isinstance(condition, tuple):
                    # AND or OR condition
                    condition_type = (
                        "AND" if any(item == "and" for item in condition) else "OR"
                    )
                    methods = [
                        m.__name__ if callable(m) else m
                        for m in condition
                        if m != "and" and m != "or"
                    ]
                    for method in methods:
                        if method not in listeners:
                            listeners[method] = []
                        listeners[method].append((attr_name, condition_type, methods))
                else:
                    raise ValueError(f"Invalid listener format for {attr_name}")

        setattr(cls, "_start_methods", start_methods)
        setattr(cls, "_listeners", listeners)

        if "initial_state" in dct:
            initial_state = dct["initial_state"]
            if isinstance(initial_state, type) and issubclass(initial_state, BaseModel):
                cls.__annotations__["state"] = initial_state
            elif isinstance(initial_state, dict):
                cls.__annotations__["state"] = Dict[str, Any]

        return cls


class Flow(Generic[T], metaclass=FlowMeta):
    _start_methods: List[str] = []
    _listeners: Dict[str, List[tuple[str, str, List[str]]]] = {}
    initial_state: Union[Type[T], T, None] = None

    def __init__(self):
        self._methods: Dict[str, Callable] = {}
        self._state = self._create_initial_state()
        self._completed_methods: set[str] = set()

        for method_name in dir(self):
            if callable(getattr(self, method_name)) and not method_name.startswith(
                "__"
            ):
                self._methods[method_name] = getattr(self, method_name)

    def _create_initial_state(self) -> T:
        if self.initial_state is None:
            return {}  # type: ignore
        elif isinstance(self.initial_state, type):
            return self.initial_state()
        else:
            return self.initial_state

    @property
    def state(self) -> T:
        return self._state

    async def kickoff(self):
        if not self._start_methods:
            raise ValueError("No start method defined")

        for start_method in self._start_methods:
            result = await self._execute_method(self._methods[start_method])
            await self._execute_listeners(start_method, result)

    async def _execute_method(self, method: Callable, *args, **kwargs):
        if inspect.iscoroutinefunction(method):
            return await method(*args, **kwargs)
        else:
            return method(*args, **kwargs)

    async def _execute_listeners(self, trigger_method: str, result: Any):
        self._completed_methods.add(trigger_method)

        if trigger_method in self._listeners:
            listener_tasks = []
            for listener, condition_type, methods in self._listeners[trigger_method]:
                if condition_type == "OR":
                    if trigger_method in methods:
                        listener_tasks.append(
                            self._execute_single_listener(listener, result)
                        )
                elif condition_type == "AND":
                    if all(method in self._completed_methods for method in methods):
                        listener_tasks.append(
                            self._execute_single_listener(listener, result)
                        )
                elif condition_type == "SINGLE":
                    listener_tasks.append(
                        self._execute_single_listener(listener, result)
                    )

            # Run all listener tasks concurrently and wait for them to complete
            await asyncio.gather(*listener_tasks)

    async def _execute_single_listener(self, listener: str, result: Any):
        try:
            method = self._methods[listener]
            sig = inspect.signature(method)
            if len(sig.parameters) > 1:  # More than just 'self'
                listener_result = await self._execute_method(method, result)
            else:
                listener_result = await self._execute_method(method)
            await self._execute_listeners(listener, listener_result)
        except Exception as e:
            print(f"Error in method {listener}: {str(e)}")


def start():
    def decorator(func):
        func.__is_start_method__ = True
        return func

    return decorator


def listen(condition):
    def decorator(func):
        func.__trigger_methods__ = condition
        return func

    return decorator
