from typing import Any, Dict, Optional

from crewai.memory.storage.interface import Storage


class Memory:
    """
    Base class for memory, now supporting agent tags and generic metadata.
    """

    def __init__(self, storage: Storage):
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

    def search(
        self,
        query: str,
        limit: int = 3,
        filters: dict = {},
        score_threshold: float = 0.35,
    ) -> Dict[str, Any]:
        return self.storage.search(
            query=query, limit=limit, filters=filters, score_threshold=score_threshold
        )
