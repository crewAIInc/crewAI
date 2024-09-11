from typing import Any, Callable, Dict, Generic, List, Type, TypeVar, Union, cast

from pydantic import BaseModel

TState = TypeVar("TState", bound=Union[BaseModel, Dict[str, Any]])


class FlowMeta(type):
    def __new__(mcs, name, bases, dct):
        cls = super().__new__(mcs, name, bases, dct)

        start_methods = []
        listeners = {}

        for attr_name, attr_value in dct.items():
            if hasattr(attr_value, "__is_start_method__"):
                start_methods.append(attr_name)
            if hasattr(attr_value, "__trigger_methods__"):
                for trigger in attr_value.__trigger_methods__:
                    trigger_name = trigger.__name__ if callable(trigger) else trigger
                    if trigger_name not in listeners:
                        listeners[trigger_name] = []
                    listeners[trigger_name].append(attr_name)

        setattr(cls, "_start_methods", start_methods)
        setattr(cls, "_listeners", listeners)

        return cls


class Flow(Generic[TState], metaclass=FlowMeta):
    _start_methods: List[str] = []
    _listeners: Dict[str, List[str]] = {}
    state_class: Type[TState]  # Class-level state_class defined once in the subclass

    def __init__(self):
        self._methods: Dict[str, Callable] = {}
        self._state: TState = self._create_default_state()

        for method_name in dir(self):
            if callable(getattr(self, method_name)) and not method_name.startswith(
                "__"
            ):
                self._methods[method_name] = getattr(self, method_name)

    @property
    def state(self) -> TState:
        """Ensure state has the correct type."""
        return self._state

    def _create_default_state(self) -> TState:
        if not hasattr(self, "state_class"):
            raise AttributeError("state_class must be defined in the Flow subclass")

        if issubclass(self.state_class, BaseModel):
            return self.state_class()  # Automatically initialize with Pydantic defaults
        elif self.state_class is dict:
            return cast(TState, DictWrapper())  # Cast to TState for DictWrapper
        else:
            raise TypeError(f"Unsupported state type: {self.state_class}")

    def run(self):
        if not self._start_methods:
            raise ValueError("No start method defined")

        for start_method in self._start_methods:
            result = self._methods[start_method]()
            self._execute_listeners(start_method, result)

    def _execute_listeners(self, trigger_method: str, result: Any):
        if trigger_method in self._listeners:
            for listener in self._listeners[trigger_method]:
                try:
                    listener_result = self._methods[listener](result)
                    self._execute_listeners(listener, listener_result)
                except Exception as e:
                    print(f"Error in method {listener}: {str(e)}")
                    return


class DictWrapper(Dict[str, Any]):
    def __getattr__(self, name: str) -> Any:
        return self.get(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def start():
    def decorator(func):
        func.__is_start_method__ = True
        return func

    return decorator


def listen(*trigger_methods):
    def decorator(func):
        func.__trigger_methods__ = trigger_methods
        return func

    return decorator
