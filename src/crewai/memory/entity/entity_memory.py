from crewai.memory.entity.entity_memory_item import EntityMemoryItem
from crewai.memory.memory import Memory
from crewai.memory.storage.rag_storage import RAGStorage


class EntityMemory(Memory):
    """
    EntityMemory class for managing structured information about entities
    and their relationships using SQLite storage.
    Inherits from the Memory class.
    """

    def __init__(self, embedder_config=None):
        storage = RAGStorage(
            type="entities", allow_reset=False, embedder_config=embedder_config
        )
        super().__init__(storage)

    def save(self, item: EntityMemoryItem) -> None:
        """Saves an entity item into the SQLite storage."""
        data = f"{item.name}({item.type}): {item.description}"
        super().save(data, item.metadata)
