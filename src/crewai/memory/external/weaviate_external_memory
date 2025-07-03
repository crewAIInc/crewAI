from typing import TYPE_CHECKING, Any, Dict, Optional
import time

from crewai.memory.external.external_memory_item import ExternalMemoryItem
from crewai.memory.memory import Memory
from crewai.memory.storage.interface import Storage
from crewai.utilities.events.crewai_event_bus import crewai_event_bus
from crewai.utilities.events.memory_events import (
    MemoryQueryStartedEvent,
    MemoryQueryCompletedEvent,
    MemoryQueryFailedEvent,
    MemorySaveStartedEvent,
    MemorySaveCompletedEvent,
    MemorySaveFailedEvent,
)

if TYPE_CHECKING:
    from crewai.memory.storage.mem0_storage import Mem0Storage


class ExternalMemory(Memory):
    def __init__(self, storage: Optional[Storage] = None, **data: Any):
        super().__init__(storage=storage, **data)

    @staticmethod
    def _configure_mem0(crew: Any, config: Dict[str, Any]) -> "Mem0Storage":
        from crewai.memory.storage.mem0_storage import Mem0Storage

        return Mem0Storage(type="external", crew=crew, config=config)

    @staticmethod
    def external_supported_storages() -> Dict[str, Any]:
        return {
            "mem0": ExternalMemory._configure_mem0,
        }

    @staticmethod
    def create_storage(crew: Any, embedder_config: Optional[Dict[str, Any]]) -> Storage:
        if not embedder_config:
            raise ValueError("embedder_config is required")

        if "provider" not in embedder_config:
            raise ValueError("embedder_config must include a 'provider' key")

        provider = embedder_config["provider"]
        supported_storages = ExternalMemory.external_supported_storages()
        if provider not in supported_storages:
            raise ValueError(f"Provider {provider} not supported")

        return supported_storages[provider](crew, embedder_config.get("config", {}))

    def save(
        self,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
        agent: Optional[str] = None,
    ) -> None:
        """Saves a value into the external storage."""
        crewai_event_bus.emit(
            self,
            event=MemorySaveStartedEvent(
                value=value,
                metadata=metadata,
                agent_role=agent,
                source_type="external_memory",
            ),
        )

        start_time = time.time()
        try:
            item = ExternalMemoryItem(value=value, metadata=metadata, agent=agent)
            super().save(value=item.value, metadata=item.metadata, agent=item.agent)

            crewai_event_bus.emit(
                self,
                event=MemorySaveCompletedEvent(
                    value=value,
                    metadata=metadata,
                    agent_role=agent,
                    save_time_ms=(time.time() - start_time) * 1000,
                    source_type="external_memory",
                ),
            )
        except Exception as e:
            crewai_event_bus.emit(
                self,
                event=MemorySaveFailedEvent(
                    value=value,
                    metadata=metadata,
                    agent_role=agent,
                    error=str(e),
                    source_type="external_memory",
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
                source_type="external_memory",
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

    def set_crew(self, crew: Any) -> "ExternalMemory":
        super().set_crew(crew)

        if not self.storage:
            self.storage = self.create_storage(crew, self.embedder_config)

        return self
