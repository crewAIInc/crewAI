from typing import Any, Dict, Optional

from crewai.memory.memory import Memory
from crewai.memory.storage.rag_storage import RAGStorage


class UserMemory(Memory):
    """
    UserMemory class for handling user memory storage and retrieval.
    Inherits from the Memory class and utilizes an instance of a class that
    adheres to the Storage for data storage, specifically working with
    MemoryItem instances.
    """

    def __init__(self, crew=None, embedder_config=None, storage=None, path=None, memory_config=None):
        # Get memory provider from crew or directly from memory_config
        memory_provider = None
        if hasattr(crew, "memory_config") and crew.memory_config is not None:
            memory_provider = crew.memory_config.get("provider")
        elif memory_config is not None:
            memory_provider = memory_config.get("provider")

        if memory_provider == "mem0":
            try:
                from crewai.memory.storage.mem0_storage import Mem0Storage
            except ImportError:
                raise ImportError(
                    "Mem0 is not installed. Please install it with `pip install mem0ai`."
                )
            storage = Mem0Storage(type="user", crew=crew)
        else:
            storage = (
                storage
                if storage
                else RAGStorage(
                    type="user",
                    allow_reset=True,
                    embedder_config=embedder_config,
                    crew=crew,
                    path=path,
                )
            )
        super().__init__(storage)

    def save(
        self,
        value,
        metadata: Optional[Dict[str, Any]] = None,
        agent: Optional[str] = None,
    ) -> None:
        if self._is_mem0_storage():
            data = f"Remember the details about the user: {value}"
        else:
            data = value
        super().save(data, metadata)
        
    def _is_mem0_storage(self) -> bool:
        """Check if the storage is Mem0Storage by checking its class name."""
        return hasattr(self.storage, "__class__") and self.storage.__class__.__name__ == "Mem0Storage"

    def search(
        self,
        query: str,
        limit: int = 3,
        score_threshold: float = 0.35,
    ):
        results = self.storage.search(
            query=query,
            limit=limit,
            score_threshold=score_threshold,
        )
        return results
