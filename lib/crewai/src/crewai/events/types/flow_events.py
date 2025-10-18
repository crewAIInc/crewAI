from typing import Any

from pydantic import BaseModel, ConfigDict

from crewai.events.base_events import BaseEvent


class FlowEvent(BaseEvent):
    """Base class for all flow events"""

    type: str
    flow_name: str


class FlowStartedEvent(FlowEvent):
    """Event emitted when a flow starts execution"""

    flow_name: str
    inputs: dict[str, Any] | None = None
    type: str = "flow_started"


class FlowCreatedEvent(FlowEvent):
    """Event emitted when a flow is created"""

    flow_name: str
    type: str = "flow_created"


class MethodExecutionStartedEvent(FlowEvent):
    """Event emitted when a flow method starts execution"""

    flow_name: str
    method_name: str
    state: dict[str, Any] | BaseModel
    params: dict[str, Any] | None = None
    type: str = "method_execution_started"


class MethodExecutionFinishedEvent(FlowEvent):
    """Event emitted when a flow method completes execution"""

    flow_name: str
    method_name: str
    result: Any = None
    state: dict[str, Any] | BaseModel
    type: str = "method_execution_finished"


class MethodExecutionFailedEvent(FlowEvent):
    """Event emitted when a flow method fails execution"""

    flow_name: str
    method_name: str
    error: Exception
    type: str = "method_execution_failed"

    model_config = ConfigDict(arbitrary_types_allowed=True)


class FlowFinishedEvent(FlowEvent):
    """Event emitted when a flow completes execution"""

    flow_name: str
    result: Any | None = None
    type: str = "flow_finished"


class FlowPlotEvent(FlowEvent):
    """Event emitted when a flow plot is created"""

    flow_name: str
    type: str = "flow_plot"
