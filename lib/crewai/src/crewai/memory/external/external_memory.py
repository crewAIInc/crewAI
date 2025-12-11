from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.memory_events import (
    MemoryQueryCompletedEvent,
    MemoryQueryFailedEvent,
    MemoryQueryStartedEvent,
    MemorySaveCompletedEvent,
    MemorySaveFailedEvent,
    MemorySaveStartedEvent,
)
from crewai.memory.external.external_memory_item import ExternalMemoryItem
from crewai.memory.memory import Memory
from crewai.memory.storage.interface import Storage
from crewai.rag.embeddings.types import ProviderSpec


if TYPE_CHECKING:
    from crewai.memory.storage.mem0_storage import Mem0Storage


class ExternalMemory(Memory):
    def __init__(self, storage: Storage | None = None, **data: Any):
        super().__init__(storage=storage, **data)

    @staticmethod
    def _configure_mem0(crew: Any, config: dict[str, Any]) -> Mem0Storage:
        from crewai.memory.storage.mem0_storage import Mem0Storage

        return Mem0Storage(type="external", crew=crew, config=config)  # type: ignore[no-untyped-call]

    @staticmethod
    def external_supported_storages() -> dict[str, Any]:
        return {
            "mem0": ExternalMemory._configure_mem0,
        }

    @staticmethod
    def create_storage(
        crew: Any, embedder_config: dict[str, Any] | ProviderSpec | None
    ) -> Storage:
        if not embedder_config:
            raise ValueError("embedder_config is required")

        if "provider" not in embedder_config:
            raise ValueError("embedder_config must include a 'provider' key")

        provider = embedder_config["provider"]
        supported_storages = ExternalMemory.external_supported_storages()
        if provider not in supported_storages:
            raise ValueError(f"Provider {provider} not supported")

        storage: Storage = supported_storages[provider](
            crew, embedder_config.get("config", {})
        )
        return storage

    def save(
        self,
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Saves a value into the external storage."""
        crewai_event_bus.emit(
            self,
            event=MemorySaveStartedEvent(
                value=value,
                metadata=metadata,
                source_type="external_memory",
                from_agent=self.agent,
                from_task=self.task,
            ),
        )

        start_time = time.time()
        try:
            item = ExternalMemoryItem(
                value=value,
                metadata=metadata,
                agent=self.agent.role if self.agent else None,
            )
            super().save(value=item.value, metadata=item.metadata)

            crewai_event_bus.emit(
                self,
                event=MemorySaveCompletedEvent(
                    value=value,
                    metadata=metadata,
                    save_time_ms=(time.time() - start_time) * 1000,
                    source_type="external_memory",
                    from_agent=self.agent,
                    from_task=self.task,
                ),
            )
        except Exception as e:
            crewai_event_bus.emit(
                self,
                event=MemorySaveFailedEvent(
                    value=value,
                    metadata=metadata,
                    error=str(e),
                    source_type="external_memory",
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
        """Search external memory for relevant entries.

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
                source_type="external_memory",
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
                    source_type="external_memory",
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
                    source_type="external_memory",
                ),
            )
            raise

    async def asave(
        self,
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save a value to external memory asynchronously.

        Args:
            value: The value to save.
            metadata: Optional metadata to associate with the value.
        """
        crewai_event_bus.emit(
            self,
            event=MemorySaveStartedEvent(
                value=value,
                metadata=metadata,
                source_type="external_memory",
                from_agent=self.agent,
                from_task=self.task,
            ),
        )

        start_time = time.time()
        try:
            item = ExternalMemoryItem(
                value=value,
                metadata=metadata,
                agent=self.agent.role if self.agent else None,
            )
            await super().asave(value=item.value, metadata=item.metadata)

            crewai_event_bus.emit(
                self,
                event=MemorySaveCompletedEvent(
                    value=value,
                    metadata=metadata,
                    save_time_ms=(time.time() - start_time) * 1000,
                    source_type="external_memory",
                    from_agent=self.agent,
                    from_task=self.task,
                ),
            )
        except Exception as e:
            crewai_event_bus.emit(
                self,
                event=MemorySaveFailedEvent(
                    value=value,
                    metadata=metadata,
                    error=str(e),
                    source_type="external_memory",
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
        """Search external memory asynchronously.

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
                source_type="external_memory",
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
                    source_type="external_memory",
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
                    source_type="external_memory",
                ),
            )
            raise

    def reset(self) -> None:
        self.storage.reset()

    def set_crew(self, crew: Any) -> ExternalMemory:
        super().set_crew(crew)

        if not self.storage:
            self.storage = self.create_storage(crew, self.embedder_config)  # type: ignore[arg-type]

        return self
