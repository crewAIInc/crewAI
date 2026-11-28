"""#8818: sync handlers registered as functools.partial must not crash the bus."""

import functools

from crewai.events.event_bus import CrewAIEventsBus
from crewai.events.types.crew_events import CrewKickoffStartedEvent


def test_sync_partial_handler_error_is_logged_not_raised() -> None:
    """A failing partial handler logs its error and emit() returns normally.

    functools.partial has no __name__, so formatting the failure message used
    to raise AttributeError inside _call_handlers, replacing the original
    handler error.
    """
    bus = CrewAIEventsBus()
    event = CrewKickoffStartedEvent(crew_name="c", inputs={})

    def boom(source, event) -> None:
        raise ValueError("handler exploded")

    bus.register_handler(CrewKickoffStartedEvent, functools.partial(boom))

    future = bus.emit(object(), event)

    assert future.result(timeout=5) is None
