from abc import ABC, abstractmethod
from logging import Logger

from crewai.utilities.events.event_bus import EventBus, event_bus


class BaseEventListener(ABC):
    def __init__(self):
        super().__init__()
        self.setup_listeners(event_bus)

    @abstractmethod
    def setup_listeners(self, event_bus: EventBus):
        pass
