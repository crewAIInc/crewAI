from typing import Any, Callable, Generic, List, Dict, Type, TypeVar
from functools import wraps
from pydantic import BaseModel


T = TypeVar("T")
EVT = TypeVar("EVT", bound=BaseModel)


class Emitter(Generic[T, EVT]):
    _listeners: Dict[Type[EVT], List[Callable]] = {}

    def on(self, event_type: Type[EVT]):
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            self._listeners.setdefault(event_type, []).append(wrapper)
            return wrapper

        return decorator

    def emit(self, source: T, event: EVT) -> None:
        event_type = type(event)
        for func in self._listeners.get(event_type, []):
            func(source, event)


default_emitter = Emitter[Any, BaseModel]()


def emit(source: Any, event: BaseModel, raise_on_error: bool = False) -> None:
    try:
        default_emitter.emit(source, event)
    except Exception as e:
        if raise_on_error:
            raise e
        else:
            print(f"Error emitting event: {e}")


def on(event_type: Type[BaseModel]) -> Callable:
    return default_emitter.on(event_type)
