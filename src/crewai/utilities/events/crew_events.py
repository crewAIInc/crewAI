from typing import Any, Dict, Optional, Union

from pydantic import InstanceOf

from crewai.utilities.events.base_events import CrewEvent


class CrewKickoffStartedEvent(CrewEvent):
    """Event emitted when a crew starts execution"""

    crew_name: Optional[str]
    inputs: Optional[Dict[str, Any]]
    type: str = "crew_kickoff_started"
    crew: Optional[Any] = None

    def __init__(self, **data):
        super().__init__(**data)
        # Set fingerprint data from the crew
        if self.crew and hasattr(self.crew, 'fingerprint') and self.crew.fingerprint:
            self.source_fingerprint = self.crew.fingerprint.uuid_str
            self.source_type = "crew"
            if hasattr(self.crew.fingerprint, 'metadata') and self.crew.fingerprint.metadata:
                self.fingerprint_metadata = self.crew.fingerprint.metadata


class CrewKickoffCompletedEvent(CrewEvent):
    """Event emitted when a crew completes execution"""

    crew_name: Optional[str]
    output: Any
    type: str = "crew_kickoff_completed"
    crew: Optional[Any] = None

    def __init__(self, **data):
        super().__init__(**data)
        # Set fingerprint data from the crew
        if self.crew and hasattr(self.crew, 'fingerprint') and self.crew.fingerprint:
            self.source_fingerprint = self.crew.fingerprint.uuid_str
            self.source_type = "crew"
            if hasattr(self.crew.fingerprint, 'metadata') and self.crew.fingerprint.metadata:
                self.fingerprint_metadata = self.crew.fingerprint.metadata


class CrewKickoffFailedEvent(CrewEvent):
    """Event emitted when a crew fails to complete execution"""

    error: str
    crew_name: Optional[str]
    type: str = "crew_kickoff_failed"
    crew: Optional[Any] = None

    def __init__(self, **data):
        super().__init__(**data)
        # Set fingerprint data from the crew
        if self.crew and hasattr(self.crew, 'fingerprint') and self.crew.fingerprint:
            self.source_fingerprint = self.crew.fingerprint.uuid_str
            self.source_type = "crew"
            if hasattr(self.crew.fingerprint, 'metadata') and self.crew.fingerprint.metadata:
                self.fingerprint_metadata = self.crew.fingerprint.metadata


class CrewTrainStartedEvent(CrewEvent):
    """Event emitted when a crew starts training"""

    crew_name: Optional[str]
    n_iterations: int
    filename: str
    inputs: Optional[Dict[str, Any]]
    type: str = "crew_train_started"
    crew: Optional[Any] = None

    def __init__(self, **data):
        super().__init__(**data)
        # Set fingerprint data from the crew
        if self.crew and hasattr(self.crew, 'fingerprint') and self.crew.fingerprint:
            self.source_fingerprint = self.crew.fingerprint.uuid_str
            self.source_type = "crew"
            if hasattr(self.crew.fingerprint, 'metadata') and self.crew.fingerprint.metadata:
                self.fingerprint_metadata = self.crew.fingerprint.metadata


class CrewTrainCompletedEvent(CrewEvent):
    """Event emitted when a crew completes training"""

    crew_name: Optional[str]
    n_iterations: int
    filename: str
    type: str = "crew_train_completed"
    crew: Optional[Any] = None

    def __init__(self, **data):
        super().__init__(**data)
        # Set fingerprint data from the crew
        if self.crew and hasattr(self.crew, 'fingerprint') and self.crew.fingerprint:
            self.source_fingerprint = self.crew.fingerprint.uuid_str
            self.source_type = "crew"
            if hasattr(self.crew.fingerprint, 'metadata') and self.crew.fingerprint.metadata:
                self.fingerprint_metadata = self.crew.fingerprint.metadata


class CrewTrainFailedEvent(CrewEvent):
    """Event emitted when a crew fails to complete training"""

    error: str
    crew_name: Optional[str]
    type: str = "crew_train_failed"
    crew: Optional[Any] = None

    def __init__(self, **data):
        super().__init__(**data)
        # Set fingerprint data from the crew
        if self.crew and hasattr(self.crew, 'fingerprint') and self.crew.fingerprint:
            self.source_fingerprint = self.crew.fingerprint.uuid_str
            self.source_type = "crew"
            if hasattr(self.crew.fingerprint, 'metadata') and self.crew.fingerprint.metadata:
                self.fingerprint_metadata = self.crew.fingerprint.metadata


class CrewTestStartedEvent(CrewEvent):
    """Event emitted when a crew starts testing"""

    crew_name: Optional[str]
    n_iterations: int
    eval_llm: Optional[Union[str, Any]]
    inputs: Optional[Dict[str, Any]]
    type: str = "crew_test_started"
    crew: Optional[Any] = None

    def __init__(self, **data):
        super().__init__(**data)
        # Set fingerprint data from the crew
        if self.crew and hasattr(self.crew, 'fingerprint') and self.crew.fingerprint:
            self.source_fingerprint = self.crew.fingerprint.uuid_str
            self.source_type = "crew"
            if hasattr(self.crew.fingerprint, 'metadata') and self.crew.fingerprint.metadata:
                self.fingerprint_metadata = self.crew.fingerprint.metadata


class CrewTestCompletedEvent(CrewEvent):
    """Event emitted when a crew completes testing"""

    crew_name: Optional[str]
    type: str = "crew_test_completed"
    crew: Optional[Any] = None

    def __init__(self, **data):
        super().__init__(**data)
        # Set fingerprint data from the crew
        if self.crew and hasattr(self.crew, 'fingerprint') and self.crew.fingerprint:
            self.source_fingerprint = self.crew.fingerprint.uuid_str
            self.source_type = "crew"
            if hasattr(self.crew.fingerprint, 'metadata') and self.crew.fingerprint.metadata:
                self.fingerprint_metadata = self.crew.fingerprint.metadata


class CrewTestFailedEvent(CrewEvent):
    """Event emitted when a crew fails to complete testing"""

    error: str
    crew_name: Optional[str]
    type: str = "crew_test_failed"
    crew: Optional[Any] = None

    def __init__(self, **data):
        super().__init__(**data)
        # Set fingerprint data from the crew
        if self.crew and hasattr(self.crew, 'fingerprint') and self.crew.fingerprint:
            self.source_fingerprint = self.crew.fingerprint.uuid_str
            self.source_type = "crew"
            if hasattr(self.crew.fingerprint, 'metadata') and self.crew.fingerprint.metadata:
                self.fingerprint_metadata = self.crew.fingerprint.metadata
