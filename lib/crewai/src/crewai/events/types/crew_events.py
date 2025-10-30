from typing import TYPE_CHECKING, Any

from crewai.events.base_events import BaseEvent


if TYPE_CHECKING:
    from crewai.crew import Crew
else:
    Crew = Any


class CrewBaseEvent(BaseEvent):
    """Base class for crew events with fingerprint handling"""

    crew_name: str | None
    crew: Crew | None = None

    def __init__(self, **data):
        super().__init__(**data)
        self.set_crew_fingerprint()

    def set_crew_fingerprint(self) -> None:
        if self.crew and hasattr(self.crew, "fingerprint") and self.crew.fingerprint:
            self.source_fingerprint = self.crew.fingerprint.uuid_str
            self.source_type = "crew"
            if (
                hasattr(self.crew.fingerprint, "metadata")
                and self.crew.fingerprint.metadata
            ):
                self.fingerprint_metadata = self.crew.fingerprint.metadata

    def to_json(self, exclude: set[str] | None = None):
        if exclude is None:
            exclude = set()
        exclude.add("crew")
        return super().to_json(exclude=exclude)


class CrewKickoffStartedEvent(CrewBaseEvent):
    """Event emitted when a crew starts execution"""

    inputs: dict[str, Any] | None
    type: str = "crew_kickoff_started"


class CrewKickoffCompletedEvent(CrewBaseEvent):
    """Event emitted when a crew completes execution"""

    output: Any
    type: str = "crew_kickoff_completed"
    total_tokens: int = 0


class CrewKickoffFailedEvent(CrewBaseEvent):
    """Event emitted when a crew fails to complete execution"""

    error: str
    type: str = "crew_kickoff_failed"


class CrewTrainStartedEvent(CrewBaseEvent):
    """Event emitted when a crew starts training"""

    n_iterations: int
    filename: str
    inputs: dict[str, Any] | None
    type: str = "crew_train_started"


class CrewTrainCompletedEvent(CrewBaseEvent):
    """Event emitted when a crew completes training"""

    n_iterations: int
    filename: str
    type: str = "crew_train_completed"


class CrewTrainFailedEvent(CrewBaseEvent):
    """Event emitted when a crew fails to complete training"""

    error: str
    type: str = "crew_train_failed"


class CrewTestStartedEvent(CrewBaseEvent):
    """Event emitted when a crew starts testing"""

    n_iterations: int
    eval_llm: str | Any | None
    inputs: dict[str, Any] | None
    type: str = "crew_test_started"


class CrewTestCompletedEvent(CrewBaseEvent):
    """Event emitted when a crew completes testing"""

    type: str = "crew_test_completed"


class CrewTestFailedEvent(CrewBaseEvent):
    """Event emitted when a crew fails to complete testing"""

    error: str
    type: str = "crew_test_failed"


class CrewTestResultEvent(CrewBaseEvent):
    """Event emitted when a crew test result is available"""

    quality: float
    execution_duration: float
    model: str
    type: str = "crew_test_result"
