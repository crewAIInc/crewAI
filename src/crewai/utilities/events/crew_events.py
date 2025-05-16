from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from crewai.utilities.events.base_events import BaseEvent

if TYPE_CHECKING:
    from crewai.crew import Crew
else:
    Crew = Any


class CrewBaseEvent(BaseEvent):
    """Base class for crew events with fingerprint handling"""

    crew_name: Optional[str]
    crew: Optional[Crew] = None

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

    inputs: Optional[Dict[str, Any]]
    type: str = "crew_kickoff_started"


class CrewKickoffCompletedEvent(CrewBaseEvent):
    """Event emitted when a crew completes execution"""

    output: Any
    type: str = "crew_kickoff_completed"


class CrewKickoffFailedEvent(CrewBaseEvent):
    """Event emitted when a crew fails to complete execution"""

    error: str
    type: str = "crew_kickoff_failed"


class CrewTrainStartedEvent(CrewBaseEvent):
    """Event emitted when a crew starts training"""

    n_iterations: int
    filename: str
    inputs: Optional[Dict[str, Any]]
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
    eval_llm: Optional[Union[str, Any]]
    inputs: Optional[Dict[str, Any]]
    type: str = "crew_test_started"


class CrewTestCompletedEvent(CrewBaseEvent):
    """Event emitted when a crew completes testing"""

    type: str = "crew_test_completed"


class CrewTestFailedEvent(CrewBaseEvent):
    """Event emitted when a crew fails to complete testing"""

    error: str
    type: str = "crew_test_failed"
