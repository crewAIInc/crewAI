from typing import Any, Dict, Optional

from pydantic import PrivateAttr

from crewai.memory.entity.entity_memory_item import EntityMemoryItem
from crewai.memory.memory import Memory
from crewai.memory.storage.rag_storage import RAGStorage


class EntityMemory(Memory):
    """
    EntityMemory class for managing structured information about entities
    and their relationships using SQLite storage.
    Inherits from the Memory class.
    """

    _memory_provider: Optional[str] = PrivateAttr()

    def __init__(self, crew=None, embedder_config=None, storage=None, path=None):
        memory_provider = None
        memory_config = None
        
        if crew and hasattr(crew, "memory_config") and crew.memory_config is not None:
            memory_config = crew.memory_config
            memory_provider = memory_config.get("provider")
        
        # If no storage is provided, try to create one
        if storage is None:
            try:
                # Try to select storage using helper method
                storage = self._select_storage(
                    storage=storage,
                    memory_config=memory_config,
                    storage_type="entity",
                    crew=crew,
                    path=path,
                    default_storage_factory=lambda path, crew: RAGStorage(
                        type="entities",
                        allow_reset=True,
                        crew=crew,
                        embedder_config=embedder_config,
                        path=path,
                    )
                )
            except ValueError:
                # Fallback to default storage
                storage = RAGStorage(
                    type="entities",
                    allow_reset=True,
                    crew=crew,
                    embedder_config=embedder_config,
                    path=path,
                )
        
        # Initialize with parameters
        super().__init__(
            storage=storage,
            embedder_config=embedder_config,
            memory_provider=memory_provider
        )
        

    def save(
        self,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
        agent: Optional[str] = None,
    ) -> None:
        """Saves an entity item or value into the storage."""
        if isinstance(value, EntityMemoryItem):
            item = value
            if self.memory_provider == "mem0":
                data = f"""
                Remember details about the following entity:
                Name: {item.name}
                Type: {item.type}
                Entity Description: {item.description}
                """
            else:
                data = f"{item.name}({item.type}): {item.description}"
            super().save(data, item.metadata)
        else:
            # Handle regular value and metadata
            super().save(value, metadata, agent)

    def reset(self) -> None:
        try:
            self.storage.reset()
        except Exception as e:
            raise Exception(f"An error occurred while resetting the entity memory: {e}")
