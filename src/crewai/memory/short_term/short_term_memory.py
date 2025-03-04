from typing import Any, Dict, Optional

from pydantic import PrivateAttr

from crewai.memory.memory import Memory
from crewai.memory.short_term.short_term_memory_item import ShortTermMemoryItem
from crewai.memory.storage.rag_storage import RAGStorage


class ShortTermMemory(Memory):
    """
    ShortTermMemory class for managing transient data related to immediate tasks
    and interactions.
    Inherits from the Memory class and utilizes an instance of a class that
    adheres to the Storage for data storage, specifically working with
    MemoryItem instances.
    """

    _memory_provider: Optional[str] = PrivateAttr()

    def __init__(self, crew=None, embedder_config=None, storage=None, path=None):
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
            self.storage = self._select_storage(
                storage=storage,
                memory_config=memory_config,
                storage_type="short_term",
                crew=crew,
                path=path,
                default_storage_factory=lambda path, crew: RAGStorage(
                    type="short_term",
                    crew=crew,
                    embedder_config=embedder_config,
                    path=path,
                )
            )
        except ValueError:
            # Fallback to default storage
            self.storage = RAGStorage(
                type="short_term",
                crew=crew,
                embedder_config=embedder_config,
                path=path,
            )

    def save(
        self,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
        agent: Optional[str] = None,
    ) -> None:
        item = ShortTermMemoryItem(data=value, metadata=metadata, agent=agent)
        if self.memory_provider == "mem0":
            item.data = f"Remember the following insights from Agent run: {item.data}"

        super().save(value=item.data, metadata=item.metadata, agent=item.agent)

    def search(
        self,
        query: str,
        limit: int = 3,
        score_threshold: float = 0.35,
    ):
        return self.storage.search(
            query=query, limit=limit, score_threshold=score_threshold
        )  # type: ignore # BUG? The reference is to the parent class, but the parent class does not have this parameters

    def reset(self) -> None:
        try:
            self.storage.reset()
        except Exception as e:
            raise Exception(
                f"An error occurred while resetting the short-term memory: {e}"
            )
