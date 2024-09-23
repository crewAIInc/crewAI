from typing import Any, Dict, Optional

from crewai.memory.memory import Memory
from crewai.memory.storage.mem0_storage import Mem0Storage


class UserMemory(Memory):
    """
    UserMemory class for handling user memory storage and retrieval.
    Inherits from the Memory class and utilizes an instance of a class that
    adheres to the Storage for data storage, specifically working with
    MemoryItem instances.
    """

    def __init__(self, crew=None):
        storage = Mem0Storage(type="user", crew=crew)
        super().__init__(storage)

    def save(
        self,
        value,
        metadata: Optional[Dict[str, Any]] = None,
        agent: Optional[str] = None,
    ) -> None:
        data = f"Remember the details about the user: {value}"
        super().save(data, metadata)

    def search(
        self,
        query: str,
        limit: int = 3,
        filters: dict = {},
        score_threshold: float = 0.35,
    ):
        return super().search(
            query=query, limit=limit, filters=filters, score_threshold=score_threshold
        )
