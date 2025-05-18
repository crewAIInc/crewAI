from typing import Any, Dict, List, Optional, Union

from crewai.memory.storage.rag_storage import RAGStorage
from crewai.utilities.logger import Logger


class MemoryOperationError(Exception):
    """
    Exception raised for errors in memory operations.
    
    Attributes:
        message: Explanation of the error
        operation: The operation that failed (e.g., "save", "search")
        memory_type: The type of memory where the error occurred
    """
    
    def __init__(self, message: str, operation: str, memory_type: str):
        self.operation = operation
        self.memory_type = memory_type
        super().__init__(f"{memory_type} {operation} error: {message}")


class Memory:
    """
    Base class for memory, now supporting agent tags and generic metadata.
    
    Attributes:
        storage: The storage backend for the memory.
        memory_verbose: Whether to log memory operations.
    """

    def __init__(self, storage: RAGStorage, memory_verbose: bool = False):
        """
        Initialize a Memory instance.
        
        Args:
            storage: The storage backend for the memory.
            memory_verbose: Whether to log memory operations.
        """
        self.storage = storage
        self.memory_verbose = memory_verbose
        self._logger = Logger(verbose=memory_verbose)

    def _log_operation(self, operation: str, details: str, agent: Optional[str] = None, level: str = "info", color: str = "cyan") -> None:
        """
        Log a memory operation if memory_verbose is enabled.
        
        Args:
            operation: The type of operation (e.g., "Saving", "Searching").
            details: Details about the operation.
            agent: The agent performing the operation, if any.
            level: The log level.
            color: The color to use for the log message.
        """
        if not self.memory_verbose:
            return
        
        sanitized_details = str(details)
        if len(sanitized_details) > 100:
            sanitized_details = f"{sanitized_details[:100]}..."
            
        memory_type = self.__class__.__name__
        agent_info = f" from agent '{agent}'" if agent else ""
        self._logger.log(level, f"{memory_type}: {operation}{agent_info}: {sanitized_details}", color=color)

    def save(
        self,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
        agent: Optional[str] = None,
    ) -> None:
        """
        Save a value to memory.
        
        Args:
            value: The value to save.
            metadata: Additional metadata to store with the value.
            agent: The agent saving the value, if any.
            
        Raises:
            MemoryOperationError: If there's an error saving the value to memory.
        """
        metadata = metadata or {}
        if agent:
            metadata["agent"] = agent
        
        if self.memory_verbose:
            self._log_operation("Saving", str(value), agent)

        try:
            self.storage.save(value, metadata)
        except Exception as e:
            if self.memory_verbose:
                self._log_operation("Error saving", str(e), agent, level="error", color="red")
            raise MemoryOperationError(str(e), "save", self.__class__.__name__)

    def search(
        self,
        query: str,
        limit: int = 3,
        score_threshold: float = 0.35,
    ) -> List[Any]:
        """
        Search for values in memory.
        
        Args:
            query: The search query.
            limit: Maximum number of results to return.
            score_threshold: Minimum similarity score for results.
            
        Returns:
            A list of matching values.
            
        Raises:
            MemoryOperationError: If there's an error searching memory.
        """
        if self.memory_verbose:
            self._log_operation("Searching for", query)
            
        try:
            results = self.storage.search(
                query=query, limit=limit, score_threshold=score_threshold
            )
            
            if self.memory_verbose and results:
                self._log_operation("Found", f"{len(results)} results")
                
            return results
        except Exception as e:
            if self.memory_verbose:
                self._log_operation("Error searching", str(e), level="error", color="red")
            raise MemoryOperationError(str(e), "search", self.__class__.__name__)
