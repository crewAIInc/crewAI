from typing import Any, Dict, Optional, Union

from pydantic import BaseModel

from .crew_events import CrewEvent


class FlowEvent(CrewEvent):
    """Base class for all flow events"""

    type: str
    flow_name: str


class FlowStartedEvent(FlowEvent):
    """Event emitted when a flow starts execution"""

    flow_name: str
    inputs: Optional[Dict[str, Any]] = None
    type: str = "flow_started"


class MethodExecutionStartedEvent(FlowEvent):
    """Event emitted when a flow method starts execution"""

    flow_name: str
    method_name: str
    state: Union[Dict[str, Any], BaseModel]
    params: Optional[Dict[str, Any]] = None
    type: str = "method_execution_started"


class MethodExecutionFinishedEvent(FlowEvent):
    """Event emitted when a flow method completes execution"""

    flow_name: str
    method_name: str
    result: Any = None
    state: Union[Dict[str, Any], BaseModel]
    type: str = "method_execution_finished"


class FlowFinishedEvent(FlowEvent):
    """Event emitted when a flow completes execution"""

    flow_name: str
    result: Optional[Any] = None
    type: str = "flow_finished"
