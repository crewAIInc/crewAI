from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class Event:
    type: str
    flow_name: str
    timestamp: datetime = field(init=False)

    def __post_init__(self):
        self.timestamp = datetime.now()


@dataclass
class FlowStartedEvent(Event):
    pass


@dataclass
class MethodExecutionStartedEvent(Event):
    method_name: str


@dataclass
class MethodExecutionFinishedEvent(Event):
    method_name: str


@dataclass
class FlowFinishedEvent(Event):
    result: Optional[Any] = None
