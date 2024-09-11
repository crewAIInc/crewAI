from typing import Callable, List, Dict, Any
from functools import wraps


class Flow:
    def __init__(self):
        self._start_method = None
        self._listeners: Dict[str, List[str]] = {}
        self._methods: Dict[str, Callable] = {}

    def run(self):
        if not self._start_method:
            raise ValueError("No start method defined")

        result = self._methods[self._start_method](self)
        self._execute_listeners(self._start_method, result)

    def _execute_listeners(self, trigger_method: str, result: Any):
        if trigger_method in self._listeners:
            for listener in self._listeners[trigger_method]:
                listener_result = self._methods[listener](self, result)
                self._execute_listeners(listener, listener_result)


def start():
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not self._start_method:
                self._start_method = func.__name__
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


def listen(*trigger_methods):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            for trigger in trigger_methods:
                trigger_name = trigger.__name__ if callable(trigger) else trigger
                if trigger_name not in self._listeners:
                    self._listeners[trigger_name] = []
                self._listeners[trigger_name].append(func.__name__)
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


class FlowMeta(type):
    def __new__(mcs, name, bases, attrs):
        new_cls = super().__new__(mcs, name, bases, attrs)
        for name, method in attrs.items():
            if hasattr(method, "_is_start"):
                new_cls._start_method = name
            if hasattr(method, "_listeners"):
                for trigger in method._listeners:
                    if trigger not in new_cls._listeners:
                        new_cls._listeners[trigger] = []
                    new_cls._listeners[trigger].append(name)
            new_cls._methods[name] = method
        return new_cls


class BaseFlow(Flow, metaclass=FlowMeta):
    _start_method = None
    _listeners = {}
    _methods = {}


# Example usage:
class ExampleFlow(BaseFlow):
    @start()
    def start_method(self):
        print("Starting the flow")
        return "Start result"

    @listen(start_method)
    def second_method(self, result):
        print(f"Second method, received: {result}")
        return "Second result"

    @listen(second_method)
    def third_method(self, result):
        print(f"Third method, received: {result}")
        return "Third result"


# Run the flow
flow = ExampleFlow()
flow.run()
