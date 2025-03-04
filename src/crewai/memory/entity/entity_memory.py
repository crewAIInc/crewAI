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
        entity_storage = None
        
        if crew and hasattr(crew, "memory_config") and crew.memory_config is not None:
            memory_provider = crew.memory_config.get("provider")
            storage_config = crew.memory_config.get("storage", {})
            entity_storage = storage_config.get("entity")
            
        super().__init__(
            storage=storage,
            embedder_config=embedder_config,
            memory_provider=memory_provider
        )

        if storage:
            # Use the provided storage
            super().__init__(storage=storage, embedder_config=embedder_config)
        elif entity_storage:
            # Use the storage from memory_config
            super().__init__(storage=entity_storage, embedder_config=embedder_config)
        elif memory_provider == "mem0":
            try:
                from crewai.memory.storage.mem0_storage import Mem0Storage
            except ImportError:
                raise ImportError(
                    "Mem0 is not installed. Please install it with `pip install mem0ai`."
                )
            super().__init__(
                storage=Mem0Storage(type="entities", crew=crew),
                embedder_config=embedder_config,
            )
        else:
            # Use RAGStorage (default)
            super().__init__(
                storage=RAGStorage(
                    type="entities",
                    allow_reset=True,
                    crew=crew,
                    embedder_config=embedder_config,
                    path=path,
                ),
                embedder_config=embedder_config,
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
