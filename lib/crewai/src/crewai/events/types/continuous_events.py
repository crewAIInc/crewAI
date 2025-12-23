"""Event types for continuous operation mode."""

from datetime import datetime
from typing import TYPE_CHECKING, Any

from crewai.events.base_events import BaseEvent
from crewai.events.types.crew_events import CrewBaseEvent

if TYPE_CHECKING:
    from crewai.crew import Crew
else:
    Crew = Any


class ContinuousKickoffStartedEvent(CrewBaseEvent):
    """Event emitted when continuous mode starts."""

    agents: list[str]
    monitoring_directive: str | None = None
    type: str = "continuous_kickoff_started"


class ContinuousKickoffStoppedEvent(CrewBaseEvent):
    """Event emitted when continuous mode stops."""

    reason: str  # "user_requested", "error", "max_runtime", "force_stop"
    total_iterations: int
    runtime_seconds: float
    type: str = "continuous_kickoff_stopped"


class ContinuousAgentActionEvent(BaseEvent):
    """Event emitted for each agent action in continuous mode."""

    agent_role: str
    agent_id: str
    action_type: str  # "tool_use", "observation", "decision", "checkpoint"
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    thought: str | None = None
    iteration: int
    timestamp: datetime
    type: str = "continuous_agent_action"

    def __init__(self, **data: Any) -> None:
        if "timestamp" not in data:
            data["timestamp"] = datetime.now()
        super().__init__(**data)


class ContinuousAgentObservationEvent(BaseEvent):
    """Event emitted when agent makes an observation."""

    agent_role: str
    agent_id: str
    observation: str
    triggered_by: str | None = None  # What caused this observation
    iteration: int
    timestamp: datetime
    type: str = "continuous_agent_observation"

    def __init__(self, **data: Any) -> None:
        if "timestamp" not in data:
            data["timestamp"] = datetime.now()
        super().__init__(**data)


class ContinuousIterationCompleteEvent(BaseEvent):
    """Event emitted after each iteration cycle."""

    iteration: int
    agents_active: list[str]
    actions_taken: int
    duration_seconds: float
    timestamp: datetime
    type: str = "continuous_iteration_complete"

    def __init__(self, **data: Any) -> None:
        if "timestamp" not in data:
            data["timestamp"] = datetime.now()
        super().__init__(**data)


class ContinuousHealthCheckEvent(CrewBaseEvent):
    """Periodic health status event."""

    uptime_seconds: float
    total_iterations: int
    agents_status: dict[str, str]  # agent_role -> status
    error_count: int = 0
    recent_errors: list[str] | None = None
    timestamp: datetime
    type: str = "continuous_health_check"

    def __init__(self, **data: Any) -> None:
        if "timestamp" not in data:
            data["timestamp"] = datetime.now()
        super().__init__(**data)


class ContinuousPausedEvent(CrewBaseEvent):
    """Event emitted when continuous operation is paused."""

    iteration: int
    timestamp: datetime
    type: str = "continuous_paused"

    def __init__(self, **data: Any) -> None:
        if "timestamp" not in data:
            data["timestamp"] = datetime.now()
        super().__init__(**data)


class ContinuousResumedEvent(CrewBaseEvent):
    """Event emitted when continuous operation is resumed."""

    iteration: int
    pause_duration_seconds: float
    timestamp: datetime
    type: str = "continuous_resumed"

    def __init__(self, **data: Any) -> None:
        if "timestamp" not in data:
            data["timestamp"] = datetime.now()
        super().__init__(**data)


class ContinuousErrorEvent(CrewBaseEvent):
    """Event emitted when an error occurs during continuous operation."""

    error: str
    error_type: str
    iteration: int
    agent_role: str | None = None
    recoverable: bool = True
    timestamp: datetime
    type: str = "continuous_error"

    def __init__(self, **data: Any) -> None:
        if "timestamp" not in data:
            data["timestamp"] = datetime.now()
        super().__init__(**data)
