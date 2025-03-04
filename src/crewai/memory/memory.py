from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field

from crewai.memory.storage.interface import SearchResult, Storage

T = TypeVar('T', bound=Storage)

class Memory(BaseModel, Generic[T]):
    """
    Base class for memory, now supporting agent tags and generic metadata.
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    embedder_config: Optional[Dict[str, Any]] = None
    storage: T
    memory_provider: Optional[str] = Field(default=None, exclude=True)

    def __init__(self, storage: T, **data: Any):
        super().__init__(storage=storage, **data)

    def _select_storage(
        self,
        storage: Optional[T] = None,
        memory_config: Optional[Dict[str, Any]] = None,
        storage_type: str = "",
        crew=None,
        path: Optional[str] = None,
        default_storage_factory: Optional[Callable] = None,
    ) -> T:
        """Helper method to select the appropriate storage based on configuration"""
        # Use the provided storage if available
        if storage:
            return storage
            
        # Use storage from memory_config if available
        if memory_config and "storage" in memory_config:
            storage_config = memory_config.get("storage", {})
            if storage_type in storage_config and storage_config[storage_type]:
                return cast(T, storage_config[storage_type])
                
        # Use Mem0Storage if specified in memory_config
        if memory_config and memory_config.get("provider") == "mem0":
            try:
                from crewai.memory.storage.mem0_storage import Mem0Storage
                return cast(T, Mem0Storage(type=storage_type, crew=crew))
            except ImportError:
                raise ImportError(
                    "Mem0 is not installed. Please install it with `pip install mem0ai`."
                )
                
        # Use default storage if provided
        if default_storage_factory:
            return cast(T, default_storage_factory(path=path, crew=crew))
            
        # Fallback to empty storage
        raise ValueError(f"No storage available for {storage_type}")

    def save(
        self,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
        agent: Optional[str] = None,
    ) -> None:
        metadata = metadata or {}
        if agent:
            metadata["agent"] = agent

        if self.storage:
            self.storage.save(value, metadata)
        else:
            raise ValueError("Storage is not initialized")

    def search(
        self,
        query: str,
        limit: int = 3,
        score_threshold: float = 0.35,
    ) -> List[SearchResult]:
        if not self.storage:
            raise ValueError("Storage is not initialized")
        return self.storage.search(
            query=query, limit=limit, score_threshold=score_threshold
        )
