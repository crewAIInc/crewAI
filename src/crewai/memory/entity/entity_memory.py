from typing import Any
import time

from pydantic import PrivateAttr

from crewai.memory.entity.entity_memory_item import EntityMemoryItem
from crewai.memory.memory import Memory
from crewai.memory.storage.rag_storage import RAGStorage
from crewai.utilities.events.crewai_event_bus import crewai_event_bus
from crewai.utilities.events.memory_events import (
    MemoryQueryStartedEvent,
    MemoryQueryCompletedEvent,
    MemoryQueryFailedEvent,
    MemorySaveStartedEvent,
    MemorySaveCompletedEvent,
    MemorySaveFailedEvent,
)


class EntityMemory(Memory):
    """
    EntityMemory class for managing structured information about entities
    and their relationships using SQLite storage.
    Inherits from the Memory class.
    """

    _memory_provider: str | None = PrivateAttr()

    def __init__(self, crew=None, embedder_config=None, storage=None, path=None):
        memory_provider = embedder_config.get("provider") if embedder_config else None
        if memory_provider == "mem0":
            try:
                from crewai.memory.storage.mem0_storage import Mem0Storage
            except ImportError:
                raise ImportError(
                    "Mem0 is not installed. Please install it with `pip install mem0ai`."
                )
            config = embedder_config.get("config") if embedder_config else None
            storage = Mem0Storage(type="short_term", crew=crew, config=config)
        else:
            storage = (
                storage
                if storage
                else RAGStorage(
                    type="entities",
                    allow_reset=True,
                    embedder_config=embedder_config,
                    crew=crew,
                    path=path,
                )
            )

        super().__init__(storage=storage)
        self._memory_provider = memory_provider

    def save(
        self,
        value: EntityMemoryItem | list[EntityMemoryItem],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Saves one or more entity items into the SQLite storage.

        Args:
            value: Single EntityMemoryItem or list of EntityMemoryItems to save.
            metadata: Optional metadata dict (included for supertype compatibility but not used).

        Notes:
            The metadata parameter is included to satisfy the supertype signature but is not
            used - entity metadata is extracted from the EntityMemoryItem objects themselves.
        """

        if not value:
            return

        items = value if isinstance(value, list) else [value]
        is_batch = len(items) > 1

        metadata = {"entity_count": len(items)} if is_batch else items[0].metadata
        crewai_event_bus.emit(
            self,
            event=MemorySaveStartedEvent(
                metadata=metadata,
                source_type="entity_memory",
                from_agent=self.agent,
                from_task=self.task,
            ),
        )

        start_time = time.time()
        saved_count = 0
        errors = []

        try:
            for item in items:
                try:
                    if self._memory_provider == "mem0":
                        data = f"""
                        Remember details about the following entity:
                        Name: {item.name}
                        Type: {item.type}
                        Entity Description: {item.description}
                        """
                    else:
                        data = f"{item.name}({item.type}): {item.description}"

                    super().save(data, item.metadata)
                    saved_count += 1
                except Exception as e:
                    errors.append(f"{item.name}: {str(e)}")

            if is_batch:
                emit_value = f"Saved {saved_count} entities"
                metadata = {"entity_count": saved_count, "errors": errors}
            else:
                emit_value = f"{items[0].name}({items[0].type}): {items[0].description}"
                metadata = items[0].metadata

            crewai_event_bus.emit(
                self,
                event=MemorySaveCompletedEvent(
                    value=emit_value,
                    metadata=metadata,
                    save_time_ms=(time.time() - start_time) * 1000,
                    source_type="entity_memory",
                    from_agent=self.agent,
                    from_task=self.task,
                ),
            )

            if errors:
                raise Exception(
                    f"Partial save: {len(errors)} failed out of {len(items)}"
                )

        except Exception as e:
            fail_metadata = (
                {"entity_count": len(items), "saved": saved_count}
                if is_batch
                else items[0].metadata
            )
            crewai_event_bus.emit(
                self,
                event=MemorySaveFailedEvent(
                    metadata=fail_metadata,
                    error=str(e),
                    source_type="entity_memory",
                    from_agent=self.agent,
                    from_task=self.task,
                ),
            )
            raise

    def search(
        self,
        query: str,
        limit: int = 3,
        score_threshold: float = 0.35,
    ):
        crewai_event_bus.emit(
            self,
            event=MemoryQueryStartedEvent(
                query=query,
                limit=limit,
                score_threshold=score_threshold,
                source_type="entity_memory",
                from_agent=self.agent,
                from_task=self.task,
            ),
        )

        start_time = time.time()
        try:
            results = super().search(
                query=query, limit=limit, score_threshold=score_threshold
            )

            crewai_event_bus.emit(
                self,
                event=MemoryQueryCompletedEvent(
                    query=query,
                    results=results,
                    limit=limit,
                    score_threshold=score_threshold,
                    query_time_ms=(time.time() - start_time) * 1000,
                    source_type="entity_memory",
                    from_agent=self.agent,
                    from_task=self.task,
                ),
            )

            return results
        except Exception as e:
            crewai_event_bus.emit(
                self,
                event=MemoryQueryFailedEvent(
                    query=query,
                    limit=limit,
                    score_threshold=score_threshold,
                    error=str(e),
                    source_type="entity_memory",
                ),
            )
            raise

    def reset(self) -> None:
        try:
            self.storage.reset()
        except Exception as e:
            raise Exception(f"An error occurred while resetting the entity memory: {e}")
