from typing import Any, Dict, Optional, Union

from pydantic import BaseModel

from .base_events import CrewEvent


class FlowEvent(CrewEvent):
    """Base class for all flow events"""

    type: str
    flow_name: str


class FlowStartedEvent(FlowEvent):
    """Event emitted when a flow starts execution"""

    flow_name: str
    inputs: Optional[Dict[str, Any]] = None
    type: str = "flow_started"


class FlowCreatedEvent(FlowEvent):
    """Event emitted when a flow is created"""

    flow_name: str
    type: str = "flow_created"


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


class MethodExecutionFailedEvent(FlowEvent):
    """Event emitted when a flow method fails execution"""

    flow_name: str
    method_name: str
    error: Any
    type: str = "method_execution_failed"


class FlowFinishedEvent(FlowEvent):
    """Event emitted when a flow completes execution"""

    flow_name: str
    result: Optional[Any] = None
    type: str = "flow_finished"


class FlowPlotEvent(FlowEvent):
    """Event emitted when a flow plot is created"""

    flow_name: str
    type: str = "flow_plot"
