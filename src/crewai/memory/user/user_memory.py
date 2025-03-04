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
        user_storage = None
        
        if crew and hasattr(crew, "memory_config") and crew.memory_config is not None:
            memory_provider = crew.memory_config.get("provider")
            storage_config = crew.memory_config.get("storage", {})
            user_storage = storage_config.get("user")
            
        super().__init__(
            storage=storage,
            embedder_config=embedder_config,
            memory_provider=memory_provider
        )

        if storage:
            # Use the provided storage
            super().__init__(storage=storage, embedder_config=embedder_config)
        elif user_storage:
            # Use the storage from memory_config
            super().__init__(storage=user_storage, embedder_config=embedder_config)
        elif memory_provider == "mem0":
            try:
                from crewai.memory.storage.mem0_storage import Mem0Storage
            except ImportError:
                raise ImportError(
                    "Mem0 is not installed. Please install it with `pip install mem0ai`."
                )
            super().__init__(
                storage=Mem0Storage(type="user", crew=crew),
                embedder_config=embedder_config,
            )
        else:
            # Use RAGStorage (default)
            from crewai.memory.storage.rag_storage import RAGStorage
            super().__init__(
                storage=RAGStorage(
                    type="user",
                    crew=crew,
                    embedder_config=embedder_config,
                    path=path,
                ),
                embedder_config=embedder_config,
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
