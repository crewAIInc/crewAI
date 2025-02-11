from typing import Union

from .agent_events import (
    AgentExecutionCompleted,
    AgentExecutionError,
    AgentExecutionStarted,
)
from .crew_events import (
    CrewKickoffCompleted,
    CrewKickoffFailed,
    CrewKickoffStarted,
    CrewTestCompleted,
    CrewTestFailed,
    CrewTestStarted,
    CrewTrainCompleted,
    CrewTrainFailed,
    CrewTrainStarted,
)
from .flow_events import (
    FlowFinished,
    FlowStarted,
    MethodExecutionFinished,
    MethodExecutionStarted,
)
from .task_events import (
    TaskCompleted,
    TaskFailed,
    TaskStarted,
)
from .tool_usage_events import ToolUsageError, ToolUsageFinished

EventTypes = Union[
    CrewKickoffStarted,
    CrewKickoffCompleted,
    CrewKickoffFailed,
    CrewTestStarted,
    CrewTestCompleted,
    CrewTestFailed,
    CrewTrainStarted,
    CrewTrainCompleted,
    CrewTrainFailed,
    AgentExecutionStarted,
    AgentExecutionCompleted,
    TaskStarted,
    TaskCompleted,
    TaskFailed,
    FlowStarted,
    FlowFinished,
    MethodExecutionStarted,
    MethodExecutionFinished,
    AgentExecutionError,
    ToolUsageFinished,
    ToolUsageError,
]
