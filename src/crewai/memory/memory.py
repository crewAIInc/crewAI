from typing import Any, Dict

from crewai.memory.storage.interface import Storage


class Memory:
    """
    Base class for memory, now supporting agent tags and generic metadata.
    """

    def __init__(self, storage: Storage):
        self.storage = storage

    def save(
        self, value: Any, metadata: Dict[str, Any] = None, agent: str = None
    ) -> None:
        metadata = metadata or {}
        if agent:
            metadata["agent"] = agent
        self.storage.save(value, metadata)

    def search(self, query: str) -> Dict[str, Any]:
        return self.storage.search(query)
