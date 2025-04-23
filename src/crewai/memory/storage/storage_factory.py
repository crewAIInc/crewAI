from typing import Any, Dict, Optional, Type, cast

from crewai.memory.storage.base_rag_storage import BaseRAGStorage
from crewai.memory.storage.rag_storage import RAGStorage
from crewai.utilities.logger import Logger


class StorageFactory:
    """Factory for creating storage instances based on provider type."""

    @classmethod
    def create_storage(
        cls,
        provider: str,
        type: str,
        allow_reset: bool = True,
        embedder_config: Optional[Any] = None,
        crew: Any = None,
        path: Optional[str] = None,
        **kwargs,
    ) -> BaseRAGStorage:
        """Create a storage instance based on the provider type.

        Args:
            provider: Type of storage provider ("chromadb", "elasticsearch", "mem0").
            type: Type of memory storage (e.g., "short_term", "entity").
            allow_reset: Whether to allow resetting the storage.
            embedder_config: Configuration for the embedder.
            crew: Crew instance.
            path: Path to the storage.
            **kwargs: Additional arguments.

        Returns:
            Storage instance.
        """
        if provider == "elasticsearch":
            try:
                from crewai.memory.storage.elasticsearch_storage import ElasticsearchStorage
                return ElasticsearchStorage(
                    type=type,
                    allow_reset=allow_reset,
                    embedder_config=embedder_config,
                    crew=crew,
                    path=path,
                    **kwargs,
                )
            except ImportError:
                Logger(verbose=True).log(
                    "error",
                    "Elasticsearch is not installed. Please install it with `pip install elasticsearch`.",
                    "red",
                )
                raise ImportError(
                    "Elasticsearch is not installed. Please install it with `pip install elasticsearch`."
                )
        elif provider == "mem0":
            try:
                from crewai.memory.storage.mem0_storage import Mem0Storage
                return cast(BaseRAGStorage, Mem0Storage(type=type, crew=crew))
            except ImportError:
                Logger(verbose=True).log(
                    "error",
                    "Mem0 is not installed. Please install it with `pip install mem0ai`.",
                    "red",
                )
                raise ImportError(
                    "Mem0 is not installed. Please install it with `pip install mem0ai`."
                )
        return RAGStorage(
            type=type,
            allow_reset=allow_reset,
            embedder_config=embedder_config,
            crew=crew,
            path=path,
        )
