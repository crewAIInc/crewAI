from abc import ABC, abstractmethod
from logging import Logger

from crewai.utilities.events.crewai_event_bus import CrewAIEventsBus, crewai_event_bus


class BaseEventListener(ABC):
    verbose: bool = False

    def __init__(self):
        super().__init__()
        self.setup_listeners(crewai_event_bus)

    @abstractmethod
    def setup_listeners(self, crewai_event_bus: CrewAIEventsBus):
        pass
