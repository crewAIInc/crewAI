from typing import Any, Dict, List, Optional

from crewai.memory.long_term.long_term_memory_item import LongTermMemoryItem
from crewai.memory.memory import Memory
# Storage factory is used to create appropriate storage backend
from crewai.memory.storage.ltm_storage_factory import LTMStorageFactory


class LongTermMemory(Memory):
    """
    LongTermMemory class for managing cross runs data related to overall crew's
    execution and performance.
    Inherits from the Memory class and utilizes an instance of a class that
    adheres to the Storage for data storage, specifically working with
    LongTermMemoryItem instances.
    """

    def __init__(
        self, 
        storage=None, 
        storage_type: str = "sqlite",
        path: Optional[str] = None,
        postgres_connection_string: Optional[str] = None,
        postgres_schema: Optional[str] = None,
        postgres_table_name: Optional[str] = None,
        postgres_min_pool_size: Optional[int] = None,
        postgres_max_pool_size: Optional[int] = None,
        postgres_use_connection_pool: Optional[bool] = None,
    ):
        """
        Initialize LongTermMemory with the specified storage backend.
        
        Args:
            storage: Optional pre-configured storage instance
            storage_type: Type of storage to use ('sqlite' or 'postgres') when creating a new storage
            path: Path to SQLite database file (only used with SQLite storage)
            postgres_connection_string: Postgres connection string (only used with Postgres storage)
            postgres_schema: Postgres schema name (only used with Postgres storage)
            postgres_table_name: Postgres table name (only used with Postgres storage)
            postgres_min_pool_size: Minimum connection pool size (only used with Postgres storage)
            postgres_max_pool_size: Maximum connection pool size (only used with Postgres storage)
            postgres_use_connection_pool: Whether to use connection pooling (only used with Postgres storage)
        """
        if not storage:
            storage = LTMStorageFactory.create_storage(
                storage_type=storage_type,
                path=path,
                connection_string=postgres_connection_string,
                schema=postgres_schema,
                table_name=postgres_table_name,
                min_pool_size=postgres_min_pool_size,
                max_pool_size=postgres_max_pool_size,
                use_connection_pool=postgres_use_connection_pool,
            )
        super().__init__(storage=storage)

    def save(self, item: LongTermMemoryItem) -> None:  # type: ignore # BUG?: Signature of "save" incompatible with supertype "Memory"
        """
        Save a memory item to storage.
        
        Args:
            item: The LongTermMemoryItem to save
        """
        # Create metadata dictionary with required values
        metadata = item.metadata.copy() if item.metadata else {}
        metadata.update({
            "agent": item.agent, 
            "expected_output": item.expected_output
        })
        
        # Ensure quality is in metadata (from item.quality if available)
        if "quality" not in metadata and item.quality is not None:
            metadata["quality"] = item.quality
        
        # Check if quality is available
        if "quality" not in metadata:
            raise ValueError("Memory quality must be provided either in item.quality or item.metadata['quality']")
            
        self.storage.save(  # type: ignore # BUG?: Unexpected keyword argument "task_description","score","datetime" for "save" of "Storage"
            task_description=item.task,
            score=metadata["quality"],
            metadata=metadata,
            datetime=item.datetime,
        )

    def search(self, task: str, latest_n: int = 3) -> List[Dict[str, Any]]:  # type: ignore # signature of "search" incompatible with supertype "Memory"
        """
        Search for memory items by task.
        
        Args:
            task: The task description to search for
            latest_n: Maximum number of results to return
            
        Returns:
            List of memory items matching the search criteria
        """
        return self.storage.load(task, latest_n) or []  # type: ignore # BUG?: "Storage" has no attribute "load"

    def reset(self) -> None:
        """Reset the storage by deleting all memory items."""
        self.storage.reset()
        
    def cleanup(self) -> None:
        """
        Clean up resources and connections.
        
        This method safely handles any exceptions that might occur during cleanup,
        ensuring resources are properly released.
        """
        if hasattr(self.storage, 'close'):
            try:
                self.storage.close()
            except Exception as e:
                # Log the error but don't raise it to ensure cleanup continues
                print(f"WARNING: Error while closing memory storage: {e}")
    
    # Keep close() for backward compatibility
    def close(self) -> None:
        """
        Close any resources held by the storage.
        
        For PostgreSQL storage with connection pooling enabled, this will
        close the connection pool. For other storage types, this is a no-op.
        
        This method safely handles any exceptions that might occur during closing.
        
        Note: This method is an alias for cleanup() and is maintained for backward compatibility.
        """
        self.cleanup()
                
    def __enter__(self):
        """Support for using LongTermMemory as a context manager."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Clean up resources when exiting a context manager block.
        
        Args:
            exc_type: Exception type if an exception was raised in the context
            exc_val: Exception value if an exception was raised in the context
            exc_tb: Exception traceback if an exception was raised in the context
        """
        self.cleanup()