from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class CrewEvent(BaseModel):
    """Base class for all crew events"""

    timestamp: datetime = Field(default_factory=datetime.now)
    type: str


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
    openai_model_name: Optional[str]
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
