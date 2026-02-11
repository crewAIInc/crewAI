import time
from typing import Any

from pydantic import PrivateAttr

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.memory_events import (
    MemoryQueryCompletedEvent,
    MemoryQueryFailedEvent,
    MemoryQueryStartedEvent,
    MemorySaveCompletedEvent,
    MemorySaveFailedEvent,
    MemorySaveStartedEvent,
)
from crewai.memory.entity.entity_memory_item import EntityMemoryItem
from crewai.memory.memory import Memory
from crewai.memory.storage.rag_storage import RAGStorage


class EntityMemory(Memory):
    """
    EntityMemory class for managing structured information about entities
    and their relationships using SQLite storage.
    Inherits from the Memory class.
    """

    _memory_provider: str | None = PrivateAttr()

    def __init__(
        self,
        crew: Any = None,
        embedder_config: Any = None,
        storage: Any = None,
        path: str | None = None,
    ) -> None:
        memory_provider = None
        if embedder_config and isinstance(embedder_config, dict):
            memory_provider = embedder_config.get("provider")

        if memory_provider == "mem0":
            try:
                from crewai.memory.storage.mem0_storage import Mem0Storage
            except ImportError as e:
                raise ImportError(
                    "Mem0 is not installed. Please install it with `pip install mem0ai`."
                ) from e
            config = (
                embedder_config.get("config")
                if embedder_config and isinstance(embedder_config, dict)
                else None
            )
            storage = Mem0Storage(type="short_term", crew=crew, config=config)  # type: ignore[no-untyped-call]
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

        def save_single_item(item: EntityMemoryItem) -> tuple[bool, str | None]:
            """Save a single item and return success status."""
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

                super(EntityMemory, self).save(data, item.metadata)
                return True, None
            except Exception as e:
                return False, f"{item.name}: {e!s}"

        try:
            for item in items:
                success, error = save_single_item(item)
                if success:
                    saved_count += 1
                else:
                    errors.append(error)

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
        limit: int = 5,
        score_threshold: float = 0.6,
    ) -> list[Any]:
        """Search entity memory for relevant entries.

        Args:
            query: The search query.
            limit: Maximum number of results to return.
            score_threshold: Minimum similarity score for results.

        Returns:
            List of matching memory entries.
        """
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

    async def asave(
        self,
        value: EntityMemoryItem | list[EntityMemoryItem],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save entity items asynchronously.

        Args:
            value: Single EntityMemoryItem or list of EntityMemoryItems to save.
            metadata: Optional metadata dict (not used, for signature compatibility).
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
        errors: list[str | None] = []

        async def save_single_item(item: EntityMemoryItem) -> tuple[bool, str | None]:
            """Save a single item asynchronously."""
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

                await super(EntityMemory, self).asave(data, item.metadata)
                return True, None
            except Exception as e:
                return False, f"{item.name}: {e!s}"

        try:
            for item in items:
                success, error = await save_single_item(item)
                if success:
                    saved_count += 1
                else:
                    errors.append(error)

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

    async def asearch(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.6,
    ) -> list[Any]:
        """Search entity memory asynchronously.

        Args:
            query: The search query.
            limit: Maximum number of results to return.
            score_threshold: Minimum similarity score for results.

        Returns:
            List of matching memory entries.
        """
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
            results = await super().asearch(
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
            raise Exception(
                f"An error occurred while resetting the entity memory: {e}"
            ) from e
