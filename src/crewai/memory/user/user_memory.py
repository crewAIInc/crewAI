from typing import Any, Dict, List, Optional

from crewai.memory.memory import Memory, MemoryOperationError


class UserMemory(Memory):
    """
    UserMemory class for handling user memory storage and retrieval.
    Inherits from the Memory class and utilizes an instance of a class that
    adheres to the Storage for data storage, specifically working with
    MemoryItem instances.
    
    Attributes:
        storage: The storage backend for the memory.
        memory_verbose: Whether to log memory operations.
    """

    def __init__(self, crew=None, memory_verbose=False):
        """
        Initialize a UserMemory instance.
        
        Args:
            crew: The crew to associate with this memory.
            memory_verbose: Whether to log memory operations.
            
        Raises:
            ImportError: If Mem0 is not installed.
        """
        try:
            from crewai.memory.storage.mem0_storage import Mem0Storage
        except ImportError:
            raise ImportError(
                "Mem0 is not installed. Please install it with `pip install mem0ai`."
            )
        storage = Mem0Storage(type="user", crew=crew)
        super().__init__(storage, memory_verbose=memory_verbose)

    def save(
        self,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
        agent: Optional[str] = None,
    ) -> None:
        """
        Save user memory.
        
        Args:
            value: The value to save.
            metadata: Additional metadata to store with the value.
            agent: The agent saving the value, if any.
            
        Raises:
            MemoryOperationError: If there's an error saving to memory.
        """
        try:
            if self.memory_verbose:
                self._log_operation("Saving user memory", str(value))
                
            # TODO: Change this function since we want to take care of the case where we save memories for the usr
            data = f"Remember the details about the user: {value}"
            super().save(data, metadata)
        except Exception as e:
            if self.memory_verbose:
                self._log_operation("Error saving user memory", str(e), level="error", color="red")
            raise MemoryOperationError(str(e), "save", self.__class__.__name__)

    def search(
        self,
        query: str,
        limit: int = 3,
        score_threshold: float = 0.35,
    ) -> List[Any]:
        """
        Search for user memories.
        
        Args:
            query: The search query.
            limit: Maximum number of results to return.
            score_threshold: Minimum similarity score for results.
            
        Returns:
            A list of matching user memories.
            
        Raises:
            MemoryOperationError: If there's an error searching memory.
        """
        try:
            if self.memory_verbose:
                self._log_operation("Searching user memory", query)
                
            results = self.storage.search(
                query=query,
                limit=limit,
                score_threshold=score_threshold,
            )
            
            if self.memory_verbose and results:
                self._log_operation("Found", f"{len(results)} results")
                
            return results
        except Exception as e:
            if self.memory_verbose:
                self._log_operation("Error searching user memory", str(e), level="error", color="red")
            raise MemoryOperationError(str(e), "search", self.__class__.__name__)
