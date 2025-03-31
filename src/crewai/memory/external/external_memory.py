from typing import Any, Dict, Optional

from crewai.memory.external.external_memory_item import ExternalMemoryItem
from crewai.memory.memory import Memory
from crewai.memory.storage.interface import Storage


class ExternalMemory(Memory):
    def __init__(self, crew=None, embedder_config=None, storage=None):
        storage = (
            storage
            if storage
            else self._create_storage(crew=crew, embedder_config=embedder_config)
        )

        super().__init__(storage=storage)

    @staticmethod
    def _configure_mem0(crew, config) -> "Mem0Storage":
        from crewai.memory.storage.mem0_storage import Mem0Storage

        return Mem0Storage(type="external", crew=crew, config=config)

    @property
    def external_supported_storages(self):
        return {
            "mem0": self._configure_mem0,
        }

    def _create_storage(self, crew, embedder_config):
        if not embedder_config:
            return Storage()

        if embedder_config and "provider" not in embedder_config:
            raise ValueError("embedder_config must include a 'provider' key")

        provider = embedder_config["provider"]
        if provider not in self.external_supported_storages:
            raise ValueError(f"Provider {provider} not supported")

        return self.external_supported_storages[provider](
            crew, embedder_config.get("config", {})
        )

    def save(
        self,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
        agent: Optional[str] = None,
    ) -> None:
        """Saves a value into the external storage."""
        item = ExternalMemoryItem(value=value, metadata=metadata, agent=agent)
        super().save(value=item.value, metadata=item.metadata, agent=item.agent)

    def reset(self) -> None:
        self.storage.reset()
