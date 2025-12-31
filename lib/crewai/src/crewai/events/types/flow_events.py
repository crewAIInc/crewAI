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


class MethodExecutionPausedEvent(FlowEvent):
    """Event emitted when a flow method is paused waiting for human feedback.

    This event is emitted when a @human_feedback decorated method with an
    async provider raises HumanFeedbackPending to pause execution.

    Attributes:
        flow_name: Name of the flow that is paused.
        method_name: Name of the method waiting for feedback.
        state: Current flow state when paused.
        flow_id: Unique identifier for this flow execution.
        message: The message shown when requesting feedback.
        emit: Optional list of possible outcomes for routing.
    """

    method_name: str
    state: dict[str, Any] | BaseModel
    flow_id: str
    message: str
    emit: list[str] | None = None
    type: str = "method_execution_paused"


class FlowFinishedEvent(FlowEvent):
    """Event emitted when a flow completes execution"""

    flow_name: str
    result: Any | None = None
    type: str = "flow_finished"
    state: dict[str, Any] | BaseModel


class FlowPausedEvent(FlowEvent):
    """Event emitted when a flow is paused waiting for human feedback.

    This event is emitted when a flow is paused due to a @human_feedback
    decorated method with an async provider raising HumanFeedbackPending.

    Attributes:
        flow_name: Name of the flow that is paused.
        flow_id: Unique identifier for this flow execution.
        method_name: Name of the method waiting for feedback.
        state: Current flow state when paused.
        message: The message shown when requesting feedback.
        emit: Optional list of possible outcomes for routing.
    """

    flow_id: str
    method_name: str
    state: dict[str, Any] | BaseModel
    message: str
    emit: list[str] | None = None
    type: str = "flow_paused"


class FlowPlotEvent(FlowEvent):
    """Event emitted when a flow plot is created"""

    flow_name: str
    type: str = "flow_plot"


class HumanFeedbackRequestedEvent(FlowEvent):
    """Event emitted when human feedback is requested.

    This event is emitted when a @human_feedback decorated method
    requires input from a human reviewer.

    Attributes:
        flow_name: Name of the flow requesting feedback.
        method_name: Name of the method decorated with @human_feedback.
        output: The method output shown to the human for review.
        message: The message displayed when requesting feedback.
        emit: Optional list of possible outcomes for routing.
    """

    method_name: str
    output: Any
    message: str
    emit: list[str] | None = None
    type: str = "human_feedback_requested"


class HumanFeedbackReceivedEvent(FlowEvent):
    """Event emitted when human feedback is received.

    This event is emitted after a human provides feedback in response
    to a @human_feedback decorated method.

    Attributes:
        flow_name: Name of the flow that received feedback.
        method_name: Name of the method that received feedback.
        feedback: The raw text feedback provided by the human.
        outcome: The collapsed outcome string (if emit was specified).
    """

    method_name: str
    feedback: str
    outcome: str | None = None
    type: str = "human_feedback_received"
