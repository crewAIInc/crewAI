from enum import Enum
from typing import Any, Callable

from blinker import signal


class CrewEvents(Enum):
    CREW_START = "crew_start"
    CREW_FINISH = "crew_finish"
    CREW_FAILURE = "crew_failure"
    TASK_START = "task_start"
    TASK_FINISH = "task_finish"
    TASK_FAILURE = "task_failure"
    AGENT_ACTION = "agent_action"
    TOOL_USE = "tool_use"
    TOKEN_USAGE = "token_usage"


class CrewEventEmitter:
    def __init__(self):
        self._all_signal = signal("all")

    def on(self, event_name: CrewEvents, callback: Callable) -> None:
        if event_name == "*":
            self._all_signal.connect(callback)
        else:
            signal(event_name.value).connect(callback)

    def emit(self, event_name: CrewEvents, *args: Any, **kwargs: Any) -> None:
        signal(event_name.value).send(*args, **kwargs)
        self._all_signal.send(event_name, *args, **kwargs)


crew_events = CrewEventEmitter()


def emit(event_name: CrewEvents, *args: Any, **kwargs: Any) -> None:
    try:
        crew_events.emit(event_name, *args, **kwargs)
    except Exception as e:
        if kwargs.get("raise_on_error", False):
            raise e
        else:
            print(f"Error emitting event: {e}")
