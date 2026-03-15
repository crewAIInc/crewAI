"""System signal event types for CrewAI.

This module contains event types for system-level signals like SIGTERM,
allowing listeners to perform cleanup operations before process termination.
"""

from collections.abc import Callable
from enum import IntEnum
import signal
from typing import Annotated, Literal, TypeVar

from pydantic import Field, TypeAdapter

from crewai.events.base_events import BaseEvent


class SignalType(IntEnum):
    """Enumeration of supported system signals."""

    SIGTERM = signal.SIGTERM
    SIGINT = signal.SIGINT
    SIGHUP = getattr(signal, "SIGHUP", 1)
    SIGTSTP = getattr(signal, "SIGTSTP", 20)
    SIGCONT = getattr(signal, "SIGCONT", 18)


class SigTermEvent(BaseEvent):
    """Event emitted when SIGTERM is received."""

    type: Literal["SIGTERM"] = "SIGTERM"
    signal_number: SignalType = SignalType.SIGTERM
    reason: str | None = None


class SigIntEvent(BaseEvent):
    """Event emitted when SIGINT is received."""

    type: Literal["SIGINT"] = "SIGINT"
    signal_number: SignalType = SignalType.SIGINT
    reason: str | None = None


class SigHupEvent(BaseEvent):
    """Event emitted when SIGHUP is received."""

    type: Literal["SIGHUP"] = "SIGHUP"
    signal_number: SignalType = SignalType.SIGHUP
    reason: str | None = None


class SigTStpEvent(BaseEvent):
    """Event emitted when SIGTSTP is received.

    Note: SIGSTOP cannot be caught - it immediately suspends the process.
    """

    type: Literal["SIGTSTP"] = "SIGTSTP"
    signal_number: SignalType = SignalType.SIGTSTP
    reason: str | None = None


class SigContEvent(BaseEvent):
    """Event emitted when SIGCONT is received."""

    type: Literal["SIGCONT"] = "SIGCONT"
    signal_number: SignalType = SignalType.SIGCONT
    reason: str | None = None


SignalEvent = Annotated[
    SigTermEvent | SigIntEvent | SigHupEvent | SigTStpEvent | SigContEvent,
    Field(discriminator="type"),
]

signal_event_adapter: TypeAdapter[SignalEvent] = TypeAdapter(SignalEvent)

SIGNAL_EVENT_TYPES: tuple[type[BaseEvent], ...] = (
    SigTermEvent,
    SigIntEvent,
    SigHupEvent,
    SigTStpEvent,
    SigContEvent,
)


T = TypeVar("T", bound=Callable[[object, SignalEvent], None])


def on_signal(func: T) -> T:
    """Decorator to register a handler for all signal events.

    Args:
        func: Handler function that receives (source, event) arguments.

    Returns:
        The original function, registered for all signal event types.
    """
    from crewai.events.event_bus import crewai_event_bus

    for event_type in SIGNAL_EVENT_TYPES:
        crewai_event_bus.on(event_type)(func)
    return func
