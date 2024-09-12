import asyncio
import inspect
from typing import Any, Callable, Dict, Generic, List, Set, Type, TypeVar, Union

from pydantic import BaseModel

T = TypeVar("T", bound=Union[BaseModel, Dict[str, Any]])


def start():
    def decorator(func):
        print(f"[start decorator] Decorating start method: {func.__name__}")
        func.__is_start_method__ = True
        return func

    return decorator


def listen(condition):
    def decorator(func):
        print(
            f"[listen decorator] Decorating listener: {func.__name__} with condition: {condition}"
        )
        if isinstance(condition, str):
            func.__trigger_methods__ = [condition]
            func.__condition_type__ = "OR"
            print(
                f"[listen decorator] Set __trigger_methods__ for {func.__name__}: [{condition}] with mode: OR"
            )
        elif (
            isinstance(condition, dict)
            and "type" in condition
            and "methods" in condition
        ):
            func.__trigger_methods__ = condition["methods"]
            func.__condition_type__ = condition["type"]
            print(
                f"[listen decorator] Set __trigger_methods__ for {func.__name__}: {func.__trigger_methods__} with mode: {func.__condition_type__}"
            )
        elif callable(condition) and hasattr(condition, "__name__"):
            func.__trigger_methods__ = [condition.__name__]
            func.__condition_type__ = "OR"
            print(
                f"[listen decorator] Set __trigger_methods__ for {func.__name__}: [{condition.__name__}] with mode: OR"
            )
        else:
            raise ValueError(
                "Condition must be a method, string, or a result of or_() or and_()"
            )
        return func

    return decorator


def or_(*conditions):
    methods = []
    for condition in conditions:
        if isinstance(condition, dict) and "methods" in condition:
            methods.extend(condition["methods"])
        elif callable(condition) and hasattr(condition, "__name__"):
            methods.append(condition.__name__)
        elif isinstance(condition, str):
            methods.append(condition)
        else:
            raise ValueError("Invalid condition in or_()")
    return {"type": "OR", "methods": methods}


def and_(*conditions):
    methods = []
    for condition in conditions:
        if isinstance(condition, dict) and "methods" in condition:
            methods.extend(condition["methods"])
        elif callable(condition) and hasattr(condition, "__name__"):
            methods.append(condition.__name__)
        elif isinstance(condition, str):
            methods.append(condition)
        else:
            raise ValueError("Invalid condition in and_()")
    return {"type": "AND", "methods": methods}


class FlowMeta(type):
    def __new__(mcs, name, bases, dct):
        cls = super().__new__(mcs, name, bases, dct)

        start_methods = []
        listeners = {}

        print(f"[FlowMeta] Processing class: {name}")
        for attr_name, attr_value in dct.items():
            print(f"[FlowMeta] Checking attribute: {attr_name}")
            if hasattr(attr_value, "__is_start_method__"):
                print(f"[FlowMeta] Found start method: {attr_name}")
                start_methods.append(attr_name)
            if hasattr(attr_value, "__trigger_methods__"):
                methods = attr_value.__trigger_methods__
                condition_type = getattr(attr_value, "__condition_type__", "OR")
                print(f"[FlowMeta] Conditions for {attr_name}:", methods)
                listeners[attr_name] = (condition_type, methods)

        setattr(cls, "_start_methods", start_methods)
        setattr(cls, "_listeners", listeners)

        print("[FlowMeta] ALL LISTENERS:", listeners)
        print("[FlowMeta] START METHODS:", start_methods)

        return cls


class Flow(Generic[T], metaclass=FlowMeta):
    _start_methods: List[str] = []
    _listeners: Dict[str, tuple[str, List[str]]] = {}
    initial_state: Union[Type[T], T, None] = None

    def __init__(self):
        print("[Flow.__init__] Initializing Flow")
        self._methods: Dict[str, Callable] = {}
        self._state = self._create_initial_state()
        self._completed_methods: Set[str] = set()
        self._pending_and_listeners: Dict[str, Set[str]] = {}

        for method_name in dir(self):
            if callable(getattr(self, method_name)) and not method_name.startswith(
                "__"
            ):
                print(f"[Flow.__init__] Adding method: {method_name}")
                self._methods[method_name] = getattr(self, method_name)

        print("[Flow.__init__] All methods:", self._methods.keys())
        print("[Flow.__init__] Listeners:", self._listeners)

    def _create_initial_state(self) -> T:
        print("[Flow._create_initial_state] Creating initial state")
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
        print("[Flow.kickoff] Starting kickoff")
        if not self._start_methods:
            raise ValueError("No start method defined")

        for start_method in self._start_methods:
            print(f"[Flow.kickoff] Executing start method: {start_method}")
            result = await self._execute_method(self._methods[start_method])
            print(
                f"[Flow.kickoff] Start method {start_method} completed. Executing listeners."
            )
            await self._execute_listeners(start_method, result)

    async def _execute_method(self, method: Callable, *args, **kwargs):
        print(f"[Flow._execute_method] Executing method: {method.__name__}")
        if inspect.iscoroutinefunction(method):
            return await method(*args, **kwargs)
        else:
            return method(*args, **kwargs)

    async def _execute_listeners(self, trigger_method: str, result: Any):
        print(
            f"[Flow._execute_listeners] Executing listeners for trigger method: {trigger_method}"
        )
        listener_tasks = []
        for listener, (condition_type, methods) in self._listeners.items():
            print(
                f"[Flow._execute_listeners] Checking listener: {listener}, condition: {condition_type}, methods: {methods}"
            )
            if condition_type == "OR":
                if trigger_method in methods:
                    print(
                        f"[Flow._execute_listeners] TRIGGERING METHOD: {listener} due to trigger: {trigger_method}"
                    )
                    listener_tasks.append(
                        self._execute_single_listener(listener, result)
                    )
            elif condition_type == "AND":
                if listener not in self._pending_and_listeners:
                    self._pending_and_listeners[listener] = set()
                self._pending_and_listeners[listener].add(trigger_method)
                if set(methods) == self._pending_and_listeners[listener]:
                    print(
                        f"[Flow._execute_listeners] All conditions met for listener: {listener}. Executing."
                    )
                    listener_tasks.append(
                        self._execute_single_listener(listener, result)
                    )
                    del self._pending_and_listeners[listener]

        # Run all listener tasks concurrently and wait for them to complete
        print(
            f"[Flow._execute_listeners] Executing {len(listener_tasks)} listener tasks"
        )
        await asyncio.gather(*listener_tasks)

    async def _execute_single_listener(self, listener: str, result: Any):
        print(f"[Flow._execute_single_listener] Executing listener: {listener}")
        try:
            method = self._methods[listener]
            sig = inspect.signature(method)
            if len(sig.parameters) > 1:  # More than just 'self'
                print(
                    f"[Flow._execute_single_listener] Executing {listener} with result"
                )
                listener_result = await self._execute_method(method, result)
            else:
                print(
                    f"[Flow._execute_single_listener] Executing {listener} without result"
                )
                listener_result = await self._execute_method(method)
            print(
                f"[Flow._execute_single_listener] {listener} completed, executing its listeners"
            )
            await self._execute_listeners(listener, listener_result)
        except Exception as e:
            print(
                f"[Flow._execute_single_listener] Error in method {listener}: {str(e)}"
            )
