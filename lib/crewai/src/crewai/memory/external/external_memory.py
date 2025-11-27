from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from crewai.events.lifecycle_decorator import with_lifecycle_events
from crewai.memory.external.external_memory_item import ExternalMemoryItem
from crewai.memory.memory import Memory
from crewai.memory.storage.interface import Storage
from crewai.rag.embeddings.types import ProviderSpec


if TYPE_CHECKING:
    from crewai.crew import Crew


class ExternalMemory(Memory):
    def __init__(self, storage: Storage | None = None, **data: Any) -> None:
        super().__init__(storage=storage, **data)

    @staticmethod
    def _configure_mem0(crew: Crew, config: dict[str, Any]) -> Storage:
        from crewai.memory.storage.mem0_storage import Mem0Config, Mem0Storage

        return Mem0Storage(
            type="external", crew=crew, config=cast(Mem0Config, cast(object, config))
        )

    @staticmethod
    def external_supported_storages() -> dict[
        str, Callable[[Crew, dict[str, Any]], Storage]
    ]:
        return {
            "mem0": ExternalMemory._configure_mem0,
        }

    @staticmethod
    def create_storage(crew: Crew, embedder_config: ProviderSpec | None) -> Storage:
        if not embedder_config:
            raise ValueError("embedder_config is required")

        if "provider" not in embedder_config:
            raise ValueError("embedder_config must include a 'provider' key")

        provider = embedder_config["provider"]
        supported_storages = ExternalMemory.external_supported_storages()
        if provider not in supported_storages:
            raise ValueError(f"Provider {provider} not supported")

        config = embedder_config.get("config", {})
        return supported_storages[provider](crew, cast(dict[str, Any], config))

    @with_lifecycle_events(
        "memory_save",
        args_map={"value": "value", "metadata": "metadata"},
        context={
            "source_type": "external_memory",
            "from_agent": lambda self: self.agent,
            "from_task": lambda self: self.task,
        },
        elapsed_name="save_time_ms",
    )
    def save(
        self,
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Saves a value into the external storage."""
        item = ExternalMemoryItem(
            value=value,
            metadata=metadata,
            agent=self.agent.role if self.agent else None,
        )
        super().save(value=item.value, metadata=item.metadata)

    @with_lifecycle_events(
        "memory_query",
        args_map={
            "query": "query",
            "limit": "limit",
            "score_threshold": "score_threshold",
        },
        context={
            "source_type": "external_memory",
            "from_agent": lambda self: self.agent,
            "from_task": lambda self: self.task,
        },
        result_name="results",
        elapsed_name="query_time_ms",
    )
    def search(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.6,
    ) -> Any:
        return super().search(query=query, limit=limit, score_threshold=score_threshold)

    def reset(self) -> None:
        self.storage.reset()

    def set_crew(self, crew: Crew) -> ExternalMemory:
        super().set_crew(crew)

        if not self.storage:
            self.storage = self.create_storage(crew, self.embedder_config)  # type: ignore[arg-type]

        return self
