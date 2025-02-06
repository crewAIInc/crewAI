from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class RunType(Enum):
    KICKOFF = "kickoff"
    TEST = "test"
    TRAIN = "train"


class CrewEvent(BaseModel):
    """Base class for all crew events"""

    timestamp: datetime = Field(default_factory=datetime.now)
    type: str


class CrewKickoffStarted(CrewEvent):
    """Event emitted when a crew starts execution"""

    crew_name: Optional[str]
    inputs: Optional[Dict[str, Any]]
    type: str = "crew_kickoff_started"


class CrewKickoffCompleted(CrewEvent):
    """Event emitted when a crew completes execution"""

    crew_name: Optional[str]
    output: Any
    type: str = "crew_kickoff_completed"


class CrewKickoffFailed(CrewEvent):
    """Event emitted when a crew fails to complete execution"""

    error: str
    type: str = "crew_kickoff_failed"


class CrewTrainStarted(CrewEvent):
    """Event emitted when a crew starts training"""

    crew_name: Optional[str]
    n_iterations: int
    filename: str
    inputs: Optional[Dict[str, Any]]
    type: str = "crew_train_started"


class CrewTrainCompleted(CrewEvent):
    """Event emitted when a crew completes training"""

    crew_name: Optional[str]
    n_iterations: int
    filename: str
    type: str = "crew_train_completed"


class CrewTrainFailed(CrewEvent):
    """Event emitted when a crew fails to complete training"""

    error: str
    type: str = "crew_train_failed"


class CrewTestStarted(CrewEvent):
    """Event emitted when a crew starts testing"""

    crew_name: Optional[str]
    n_iterations: int
    openai_model_name: Optional[str]
    inputs: Optional[Dict[str, Any]]
    type: str = "crew_test_started"


class CrewTestCompleted(CrewEvent):
    """Event emitted when a crew completes testing"""

    crew_name: Optional[str]
    type: str = "crew_test_completed"


class CrewTestFailed(CrewEvent):
    """Event emitted when a crew fails to complete testing"""

    error: str
    type: str = "crew_test_failed"
