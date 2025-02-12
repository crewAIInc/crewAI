from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel


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
    state: Union[Dict[str, Any], BaseModel]
    params: Optional[Dict[str, Any]] = None


@dataclass
class MethodExecutionFinishedEvent(Event):
    method_name: str
    state: Union[Dict[str, Any], BaseModel]
    result: Any = None


@dataclass
class FlowFinishedEvent(Event):
    result: Optional[Any] = None
