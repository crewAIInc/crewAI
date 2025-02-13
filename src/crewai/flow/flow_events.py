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

    def __post_init__(self):
        super().__post_init__()
        # Create a new instance of BaseModel state to avoid pickling issues
        if isinstance(self.state, BaseModel):
            self.state = type(self.state)(**self.state.model_dump())


@dataclass
class MethodExecutionFinishedEvent(Event):
    method_name: str
    state: Union[Dict[str, Any], BaseModel]
    result: Any = None

    def __post_init__(self):
        super().__post_init__()
        # Create a new instance of BaseModel state to avoid pickling issues
        if isinstance(self.state, BaseModel):
            self.state = type(self.state)(**self.state.model_dump())


@dataclass
class FlowFinishedEvent(Event):
    result: Optional[Any] = None
