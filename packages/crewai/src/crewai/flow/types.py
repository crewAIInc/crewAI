"""Type definitions for CrewAI Flow module.

This module contains TypedDict definitions and type aliases used throughout
the Flow system.
"""

from typing import Any, TypedDict
from typing_extensions import NotRequired, Required


class FlowMethodData(TypedDict):
    """Flow method information.

    Attributes:
        name: The name of the flow method.
        starting_point: Whether this method is a starting point for the flow.
    """

    name: str
    starting_point: NotRequired[bool]


class CompletedMethodData(TypedDict):
    """Completed method information.

    Represents a flow method that has been successfully executed.

    Attributes:
        flow_method: The flow method information.
        status: The completion status of the method.
    """

    flow_method: FlowMethodData
    status: str


class ExecutionMethodData(TypedDict, total=False):
    """Execution method information.

    Contains detailed information about a method's execution, including
    timing, state, and any error details.

    Attributes:
        flow_method: The flow method information.
        started_at: ISO timestamp when the method started execution.
        finished_at: ISO timestamp when the method finished execution, if completed.
        status: Current status of the method execution.
        initial_state: The state before method execution.
        final_state: The state after method execution.
        error_details: Details about any error that occurred during execution.
    """

    flow_method: Required[FlowMethodData]
    started_at: Required[str]
    status: Required[str]
    finished_at: str
    initial_state: dict[str, Any]
    final_state: dict[str, Any]
    error_details: dict[str, Any]


class FlowData(TypedDict):
    """Flow structure information.

    Contains metadata about the flow structure and its methods.

    Attributes:
        name: The name of the flow.
        flow_methods_attributes: List of all flow methods and their attributes.
    """

    name: str
    flow_methods_attributes: list[FlowMethodData]


class FlowExecutionData(TypedDict):
    """Flow execution data.

    Complete execution data for a flow, including its current state,
    completed methods, and execution history. Used for resuming flows
    from a previous state.

    Attributes:
        id: Unique identifier for the flow execution.
        flow: Flow structure and metadata.
        inputs: Input data provided to the flow.
        completed_methods: List of methods that have been successfully completed.
        execution_methods: Detailed execution history for all methods.
    """

    id: str
    flow: FlowData
    inputs: dict[str, Any]
    completed_methods: list[CompletedMethodData]
    execution_methods: list[ExecutionMethodData]
