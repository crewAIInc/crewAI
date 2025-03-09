from typing import Any, Dict, Optional

from crewai.memory.memory import Memory


class UserMemory(Memory):
    """
    UserMemory class for handling user memory storage and retrieval.
    Inherits from the Memory class and utilizes an instance of a class that
    adheres to the Storage for data storage, specifically working with
    MemoryItem instances.
    """

    def __init__(self, crew=None, embedder_config=None, storage=None, path=None, **kwargs):
        memory_provider = None
        memory_config = None
        
        if crew and hasattr(crew, "memory_config") and crew.memory_config is not None:
            memory_config = crew.memory_config
            memory_provider = memory_config.get("provider")
            
        # Initialize with basic parameters
        super().__init__(
            storage=storage,
            embedder_config=embedder_config,
            memory_provider=memory_provider
        )
        
        try:
            # Try to select storage using helper method
            from crewai.memory.storage.rag_storage import RAGStorage
            self.storage = self._select_storage(
                storage=storage,
                memory_config=memory_config,
                storage_type="user",
                crew=crew,
                path=path,
                default_storage_factory=lambda path, crew: RAGStorage(
                    type="user",
                    crew=crew,
                    embedder_config=embedder_config,
                    path=path,
                )
            )
        except ValueError:
            # Fallback to default storage
            from crewai.memory.storage.rag_storage import RAGStorage
            self.storage = RAGStorage(
                type="user",
                crew=crew,
                embedder_config=embedder_config,
                path=path,
            )

    def save(
        self,
        value,
        metadata: Optional[Dict[str, Any]] = None,
        agent: Optional[str] = None,
    ) -> None:
        # TODO: Change this function since we want to take care of the case where we save memories for the usr
        data = f"Remember the details about the user: {value}"
        super().save(data, metadata)

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
        
    def reset(self) -> None:
        try:
            self.storage.reset()
        except Exception as e:
            raise Exception(f"An error occurred while resetting the user memory: {e}")
