from typing import TYPE_CHECKING, Any

from crewai.memory.external.external_memory_item import ExternalMemoryItem
from crewai.memory.memory import Memory
from crewai.memory.storage.interface import Storage

if TYPE_CHECKING:
    from crewai.memory.storage.mem0_storage import Mem0Storage


class ExternalMemory(Memory):
    def __init__(self, storage: Storage | None = None, **data: Any) -> None:
        super().__init__(storage=storage, **data)

    @staticmethod
    def _configure_mem0(crew: Any, config: dict[str, Any]) -> "Mem0Storage":
        from crewai.memory.storage.mem0_storage import Mem0Storage

        return Mem0Storage(type="external", crew=crew, config=config)

    @staticmethod
    def external_supported_storages() -> dict[str, Any]:
        return {
            "mem0": ExternalMemory._configure_mem0,
        }

    @staticmethod
    def create_storage(crew: Any, embedder_config: dict[str, Any] | None) -> Storage:
        if not embedder_config:
            msg = "embedder_config is required"
            raise ValueError(msg)

        if "provider" not in embedder_config:
            msg = "embedder_config must include a 'provider' key"
            raise ValueError(msg)

        provider = embedder_config["provider"]
        supported_storages = ExternalMemory.external_supported_storages()
        if provider not in supported_storages:
            msg = f"Provider {provider} not supported"
            raise ValueError(msg)

        return supported_storages[provider](crew, embedder_config.get("config", {}))

    def save(
        self,
        value: Any,
        metadata: dict[str, Any] | None = None,
        agent: str | None = None,
    ) -> None:
        """Saves a value into the external storage."""
        item = ExternalMemoryItem(value=value, metadata=metadata, agent=agent)
        super().save(value=item.value, metadata=item.metadata, agent=item.agent)

    def reset(self) -> None:
        self.storage.reset()

    def set_crew(self, crew: Any) -> "ExternalMemory":
        super().set_crew(crew)

        if not self.storage:
            self.storage = self.create_storage(crew, self.embedder_config)

        return self
