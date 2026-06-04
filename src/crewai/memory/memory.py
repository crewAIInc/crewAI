import logging
from typing import Any, Callable, Dict, List, Optional

from crewai.memory.storage.rag_storage import RAGStorage

logger = logging.getLogger(__name__)


class Memory:
    """
    Base class for memory, now supporting agent tags and generic metadata.
    """

    def __init__(self, storage: RAGStorage, memory_guard: Optional[Callable[[str], bool]] = None):
        self.storage = storage
        self.memory_guard = memory_guard

    def save(
        self,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
        agent: Optional[str] = None,
    ) -> None:
        if self.memory_guard is not None:
            if not self.memory_guard(str(value)):
                logger.warning(
                    "Memory guard blocked a memory write (agent=%s).", agent
                )
                return

        metadata = metadata or {}
        if agent:
            metadata["agent"] = agent

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
