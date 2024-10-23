from typing import Any, Dict, Optional, List

from crewai.memory.storage.rag_storage import RAGStorage


class Memory:
    """
    Base class for memory, now supporting agent tags and generic metadata.
    """

    def __init__(self, storage: RAGStorage):
        self.storage = storage

    def save(
        self,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
        agent: Optional[str] = None,
    ) -> None:
        metadata = metadata or {}
        if agent:
            metadata["agent"] = agent

        self.storage.save(value, metadata)

    def search(self, query: str) -> List[Dict[str, Any]]:
        return self.storage.search(query)
