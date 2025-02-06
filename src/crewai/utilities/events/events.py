from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Type,
    TypeVar,
    TYPE_CHECKING,
)
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field
from .event_types import EventTypes

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


# Global event emitter instance
default_emitter = Emitter[Any, BaseModel]()


def emit(source: Any, event: BaseModel) -> None:
    """Emit an event to all registered listeners"""
    default_emitter.emit(source, event)


def on(event_type: Type[EventTypes]) -> Callable:
    """Register a listener for a specific event type"""
    return default_emitter.on(event_type)
