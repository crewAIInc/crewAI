from typing import Any, Dict, List, Optional

from pydantic import PrivateAttr

from crewai.memory.memory import Memory
from crewai.memory.storage.rag_storage import RAGStorage


class UserMemory(Memory):
    """
    UserMemory class for handling user memory storage and retrieval.
    Inherits from the Memory class and utilizes an instance of a class that
    adheres to the Storage for data storage, specifically working with
    MemoryItem instances.
    """

    _memory_provider: Optional[str] = PrivateAttr()

    def __init__(
        self, 
        crew=None, 
        embedder_config: Optional[Dict[str, Any]] = None, 
        storage: Optional[Any] = None, 
        path: Optional[str] = None, 
        memory_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize UserMemory with the specified storage provider.
        
        Args:
            crew: Optional crew object that may contain memory configuration
            embedder_config: Optional configuration for the embedder
            storage: Optional pre-configured storage instance
            path: Optional path for storage
            memory_config: Optional explicit memory configuration
        """
        # Get memory provider from crew or directly from memory_config
        memory_provider = None
        if crew and hasattr(crew, "memory_config") and crew.memory_config is not None:
            memory_provider = crew.memory_config.get("provider")
        elif memory_config is not None:
            memory_provider = memory_config.get("provider")

        if memory_provider == "mem0":
            try:
                from crewai.memory.storage.mem0_storage import Mem0Storage
            except ImportError:
                raise ImportError(
                    "Mem0 is not installed. Please install it with `pip install mem0ai`."
                )
            storage = Mem0Storage(type="user", crew=crew)
        else:
            storage = (
                storage
                if storage
                else RAGStorage(
                    type="user",
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
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
        agent: Optional[str] = None,
    ) -> None:
        """
        Save user memory data with appropriate formatting based on the storage provider.
        
        Args:
            value: The data to save
            metadata: Optional metadata to associate with the memory
            agent: Optional agent name to associate with the memory
        """
        if self._memory_provider == "mem0":
            data = f"Remember the details about the user: {value}"
        else:
            data = value
        super().save(data, metadata, agent)

    def search(
        self,
        query: str,
        limit: int = 3,
        score_threshold: float = 0.35,
    ) -> List[Any]:
        """
        Search for user memories that match the query.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score for results
            
        Returns:
            List of matching memory items
        """
        return self.storage.search(
            query=query,
            limit=limit,
            score_threshold=score_threshold,
        )
        
    def reset(self) -> None:
        """Reset the user memory storage."""
        try:
            self.storage.reset()
        except Exception as e:
            raise Exception(f"An error occurred while resetting the user memory: {e}")
