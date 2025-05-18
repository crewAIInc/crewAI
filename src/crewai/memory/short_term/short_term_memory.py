from typing import Any, Dict, List, Optional

from crewai.memory.memory import Memory, MemoryOperationError
from crewai.memory.short_term.short_term_memory_item import ShortTermMemoryItem
from crewai.memory.storage.rag_storage import RAGStorage


class ShortTermMemory(Memory):
    """
    ShortTermMemory class for managing transient data related to immediate tasks
    and interactions.
    Inherits from the Memory class and utilizes an instance of a class that
    adheres to the Storage for data storage, specifically working with
    MemoryItem instances.
    
    Attributes:
        memory_provider: The memory provider to use, if any.
        storage: The storage backend for the memory.
        memory_verbose: Whether to log memory operations.
    """

    def __init__(self, crew=None, embedder_config=None, storage=None, path=None, memory_verbose=False):
        """
        Initialize a ShortTermMemory instance.
        
        Args:
            crew: The crew to associate with this memory.
            embedder_config: Configuration for the embedder.
            storage: The storage backend for the memory.
            path: Path to the storage file, if any.
            memory_verbose: Whether to log memory operations.
        """
        if hasattr(crew, "memory_config") and crew.memory_config is not None:
            self.memory_provider = crew.memory_config.get("provider")
        else:
            self.memory_provider = None

        if self.memory_provider == "mem0":
            try:
                from crewai.memory.storage.mem0_storage import Mem0Storage
            except ImportError:
                raise ImportError(
                    "Mem0 is not installed. Please install it with `pip install mem0ai`."
                )
            storage = Mem0Storage(type="short_term", crew=crew)
        else:
            storage = (
                storage
                if storage
                else RAGStorage(
                    type="short_term",
                    embedder_config=embedder_config,
                    crew=crew,
                    path=path,
                )
            )
        super().__init__(storage, memory_verbose=memory_verbose)

    def save(
        self,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
        agent: Optional[str] = None,
    ) -> None:
        """
        Save a value to short-term memory.
        
        Args:
            value: The value to save.
            metadata: Additional metadata to store with the value.
            agent: The agent saving the value, if any.
            
        Raises:
            MemoryOperationError: If there's an error saving to memory.
        """
        try:
            item = ShortTermMemoryItem(data=value, metadata=metadata, agent=agent)
            if self.memory_verbose:
                self._log_operation("Saving item", str(item.data), agent)
                
            if self.memory_provider == "mem0":
                item.data = f"Remember the following insights from Agent run: {item.data}"

            super().save(value=item.data, metadata=item.metadata, agent=item.agent)
        except Exception as e:
            if self.memory_verbose:
                self._log_operation("Error saving item", str(e), level="error", color="red")
            raise MemoryOperationError(str(e), "save", self.__class__.__name__)

    def search(
        self,
        query: str,
        limit: int = 3,
        score_threshold: float = 0.35,
    ) -> List[Any]:
        """
        Search for values in short-term memory.
        
        Args:
            query: The search query.
            limit: Maximum number of results to return.
            score_threshold: Minimum similarity score for results.
            
        Returns:
            A list of matching values.
            
        Raises:
            MemoryOperationError: If there's an error searching memory.
        """
        try:
            return super().search(query=query, limit=limit, score_threshold=score_threshold)
        except Exception as e:
            if self.memory_verbose:
                self._log_operation("Error searching", str(e), level="error", color="red")
            raise MemoryOperationError(str(e), "search", self.__class__.__name__)

    def reset(self) -> None:
        """
        Reset the short-term memory.
        
        Raises:
            MemoryOperationError: If there's an error resetting the memory.
        """
        try:
            self.storage.reset()
        except Exception as e:
            if self.memory_verbose:
                self._log_operation("Error resetting", str(e), level="error", color="red")
            raise MemoryOperationError(str(e), "reset", self.__class__.__name__)
