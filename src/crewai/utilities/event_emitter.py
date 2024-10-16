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

    def on(self, event_name: CrewEvents | str, callback: Callable) -> None:
        print("Connecting signal:", event_name)
        if event_name == "*" or event_name == "all":
            self._all_signal.connect(callback, weak=False)
            print("Connected to all_signal")
        else:
            signal(
                event_name.value if isinstance(event_name, CrewEvents) else event_name
            ).connect(callback, weak=False)

    def emit(self, event_name: CrewEvents, *args: Any, **kwargs: Any) -> None:
        print(f"Emitting signal: {event_name.value}")
        print("args", args)
        print("kwargs", kwargs)
        signal(event_name.value).send(*args, **kwargs)
        print(f"Emitting all signal for: {event_name.value}")
        self._all_signal.send(*args, event=event_name.value, **kwargs)


crew_events = CrewEventEmitter()


def emit(event_name: CrewEvents, *args: Any, **kwargs: Any) -> None:
    print("Calling emit", event_name)
    print("Args:", args)
    print("Kwargs:", kwargs)
    try:
        crew_events.emit(event_name, *args, **kwargs)
    except Exception as e:
        if kwargs.get("raise_on_error", False):
            raise e
        else:
            print(f"Error emitting event: {e}")
