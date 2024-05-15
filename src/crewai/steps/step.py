from abc import ABC, abstractmethod
from typing import List, Dict, Any


class Step(ABC):
    @abstractmethod
    def kickoff(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pass
