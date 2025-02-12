from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class Event:
    type: str
    flow_name: str
    timestamp: datetime = field(init=False)

    def __post_init__(self):
        self.timestamp = datetime.now()


@dataclass
class FlowStartedEvent(Event):
    inputs: Optional[Dict[str, Any]] = None


@dataclass
class MethodExecutionStartedEvent(Event):
    method_name: str
    params: Optional[Dict[str, Any]] = None
    state: Optional[Dict[str, Any]] = None


@dataclass
class MethodExecutionFinishedEvent(Event):
    method_name: str
    result: Any = None
    state: Optional[Dict[str, Any]] = None


@dataclass
class FlowFinishedEvent(Event):
    result: Optional[Any] = None
