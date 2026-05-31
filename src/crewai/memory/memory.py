from typing import Any, Dict, List, Optional

from crewai.memory.sanitizer import MemorySanitizer, get_default_sanitizer
from crewai.memory.storage.rag_storage import RAGStorage


class Memory:
    """
    Base class for memory, now supporting agent tags and generic metadata.
    """

    def __init__(
        self,
        storage: RAGStorage,
        sanitizer: Optional[MemorySanitizer] = None,
    ):
        self.storage = storage
        self.sanitizer = sanitizer or get_default_sanitizer()

    def save(
        self,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
        agent: Optional[str] = None,
    ) -> None:
        metadata = metadata or {}
        if agent:
            metadata["agent"] = agent

        if isinstance(value, str):
            value = self.sanitizer.sanitize(value)

        self.storage.save(value, metadata)

    def search(
        self,
        query: str,
        limit: int = 3,
        score_threshold: float = 0.35,
    ) -> List[Any]:
        return self.storage.search(
            query=query, limit=limit, score_threshold=score_threshold
        )
