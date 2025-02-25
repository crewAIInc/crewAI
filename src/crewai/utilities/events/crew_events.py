from typing import Any, Dict, Optional, Union

from pydantic import InstanceOf

from crewai.utilities.events.base_events import CrewEvent


class CrewKickoffStartedEvent(CrewEvent):
    """Event emitted when a crew starts execution"""

    crew_name: Optional[str]
    inputs: Optional[Dict[str, Any]]
    type: str = "crew_kickoff_started"


class CrewKickoffCompletedEvent(CrewEvent):
    """Event emitted when a crew completes execution"""

    crew_name: Optional[str]
    output: Any
    type: str = "crew_kickoff_completed"


class CrewKickoffFailedEvent(CrewEvent):
    """Event emitted when a crew fails to complete execution"""

    error: str
    crew_name: Optional[str]
    type: str = "crew_kickoff_failed"


class CrewTrainStartedEvent(CrewEvent):
    """Event emitted when a crew starts training"""

    crew_name: Optional[str]
    n_iterations: int
    filename: str
    inputs: Optional[Dict[str, Any]]
    type: str = "crew_train_started"


class CrewTrainCompletedEvent(CrewEvent):
    """Event emitted when a crew completes training"""

    crew_name: Optional[str]
    n_iterations: int
    filename: str
    type: str = "crew_train_completed"


class CrewTrainFailedEvent(CrewEvent):
    """Event emitted when a crew fails to complete training"""

    error: str
    crew_name: Optional[str]
    type: str = "crew_train_failed"


class CrewTestStartedEvent(CrewEvent):
    """Event emitted when a crew starts testing"""

    crew_name: Optional[str]
    n_iterations: int
    eval_llm: Optional[Union[str, Any]]
    inputs: Optional[Dict[str, Any]]
    type: str = "crew_test_started"


class CrewTestCompletedEvent(CrewEvent):
    """Event emitted when a crew completes testing"""

    crew_name: Optional[str]
    type: str = "crew_test_completed"


class CrewTestFailedEvent(CrewEvent):
    """Event emitted when a crew fails to complete testing"""

    error: str
    crew_name: Optional[str]
    type: str = "crew_test_failed"
