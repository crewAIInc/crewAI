from typing import Any, Dict, List, Optional

from crewai.memory.long_term.long_term_memory_item import LongTermMemoryItem
from crewai.memory.memory import Memory, MemoryOperationError
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage


class LongTermMemory(Memory):
    """
    LongTermMemory class for managing cross runs data related to overall crew's
    execution and performance.
    Inherits from the Memory class and utilizes an instance of a class that
    adheres to the Storage for data storage, specifically working with
    LongTermMemoryItem instances.
    
    Attributes:
        storage: The storage backend for the memory.
        memory_verbose: Whether to log memory operations.
    """

    def __init__(self, storage=None, path=None, memory_verbose=False):
        """
        Initialize a LongTermMemory instance.
        
        Args:
            storage: The storage backend for the memory.
            path: Path to the storage file, if any.
            memory_verbose: Whether to log memory operations.
        """
        if not storage:
            storage = LTMSQLiteStorage(db_path=path) if path else LTMSQLiteStorage()
        super().__init__(storage, memory_verbose=memory_verbose)

    def save(self, item: LongTermMemoryItem) -> None:  # type: ignore # BUG?: Signature of "save" incompatible with supertype "Memory"
        """
        Save a long-term memory item to storage.
        
        Args:
            item: The long-term memory item to save.
            
        Raises:
            MemoryOperationError: If there's an error saving the item to memory.
        """
        try:
            if self.memory_verbose:
                self._log_operation("Saving task", item.task)
                self._log_operation("Agent", item.agent)
                self._log_operation("Quality", str(item.metadata.get('quality')))
                
            metadata = item.metadata
            metadata.update({"agent": item.agent, "expected_output": item.expected_output})
            self.storage.save(  # type: ignore # BUG?: Unexpected keyword argument "task_description","score","datetime" for "save" of "Storage"
                task_description=item.task,
                score=metadata["quality"],
                metadata=metadata,
                datetime=item.datetime,
            )
        except Exception as e:
            if self.memory_verbose:
                self._log_operation("Error saving task", str(e), level="error", color="red")
            raise MemoryOperationError(str(e), "save task", self.__class__.__name__)

    def search(self, task: str, latest_n: int = 3) -> List[Dict[str, Any]]:  # type: ignore # signature of "search" incompatible with supertype "Memory"
        """
        Search for long-term memories related to a task.
        
        Args:
            task: The task description to search for.
            latest_n: Maximum number of results to return.
            
        Returns:
            A list of matching long-term memories.
            
        Raises:
            MemoryOperationError: If there's an error searching memory.
        """
        try:
            if self.memory_verbose:
                self._log_operation("Searching for task", task)
            results = self.storage.load(task, latest_n)  # type: ignore # BUG?: "Storage" has no attribute "load"
            if self.memory_verbose and results:
                self._log_operation("Found", f"{len(results)} results")
            return results
        except Exception as e:
            if self.memory_verbose:
                self._log_operation("Error searching", str(e), level="error", color="red")
            raise MemoryOperationError(str(e), "search", self.__class__.__name__)

    def reset(self) -> None:
        """
        Reset the long-term memory.
        
        Raises:
            MemoryOperationError: If there's an error resetting the memory.
        """
        try:
            self.storage.reset()
        except Exception as e:
            if self.memory_verbose:
                self._log_operation("Error resetting", str(e), level="error", color="red")
            raise MemoryOperationError(str(e), "reset", self.__class__.__name__)
