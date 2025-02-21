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
class BaseStateEvent(Event):
    """Base class for events containing state data.
    
    Handles common state serialization and validation logic to ensure thread-safe
    state handling and proper type validation.
    
    Raises:
        ValueError: If state has invalid type
    """
    state: Union[Dict[str, Any], BaseModel]

    def __post_init__(self):
        super().__post_init__()
        self._process_state()
    
    def _process_state(self):
        """Process and validate state data.
        
        Ensures state is of valid type and creates a new instance of BaseModel
        states to avoid thread lock serialization issues.
        
        Raises:
            ValueError: If state has invalid type
        """
        if not isinstance(self.state, (dict, BaseModel)):
            raise ValueError(f"Invalid state type: {type(self.state)}")
        if isinstance(self.state, BaseModel):
            self.state = type(self.state)(**self.state.model_dump())


@dataclass
class FlowStartedEvent(Event):
    inputs: Optional[Dict[str, Any]] = None


@dataclass
class MethodExecutionStartedEvent(BaseStateEvent):
    method_name: str
    state: Union[Dict[str, Any], BaseModel]
    params: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        super().__post_init__()
        self._process_state()


@dataclass
class MethodExecutionFinishedEvent(BaseStateEvent):
    method_name: str
    state: Union[Dict[str, Any], BaseModel]
    result: Any = None

    def __post_init__(self):
        super().__post_init__()
        self._process_state()


@dataclass
class FlowFinishedEvent(Event):
    result: Optional[Any] = None
