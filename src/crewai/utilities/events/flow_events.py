from typing import Any, Dict, Optional

from .crew_events import CrewEvent


class FlowStarted(CrewEvent):
    """Event emitted when a flow starts execution"""

    flow_name: str
    type: str = "flow_started"


class MethodExecutionStarted(CrewEvent):
    """Event emitted when a flow method starts execution"""

    flow_name: str
    method_name: str
    type: str = "method_execution_started"


class MethodExecutionFinished(CrewEvent):
    """Event emitted when a flow method completes execution"""

    flow_name: str
    method_name: str
    type: str = "method_execution_finished"


class FlowFinished(CrewEvent):
    """Event emitted when a flow completes execution"""

    flow_name: str
    result: Optional[Any] = None
    type: str = "flow_finished"
