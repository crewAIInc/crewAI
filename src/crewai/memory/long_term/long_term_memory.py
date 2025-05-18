from typing import Any, Dict, List

from crewai.memory.long_term.long_term_memory_item import LongTermMemoryItem
from crewai.memory.memory import Memory
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage


class LongTermMemory(Memory):
    """
    LongTermMemory class for managing cross runs data related to overall crew's
    execution and performance.
    Inherits from the Memory class and utilizes an instance of a class that
    adheres to the Storage for data storage, specifically working with
    LongTermMemoryItem instances.
    """

    def __init__(self, storage=None, path=None, memory_verbose=False):
        if not storage:
            storage = LTMSQLiteStorage(db_path=path) if path else LTMSQLiteStorage()
        super().__init__(storage, memory_verbose=memory_verbose)

    def save(self, item: LongTermMemoryItem) -> None:  # type: ignore # BUG?: Signature of "save" incompatible with supertype "Memory"
        if self.memory_verbose:
            memory_type = self.__class__.__name__
            self._logger.log("info", f"{memory_type}: Saving task: {item.task[:100]}{'...' if len(item.task) > 100 else ''}", color="cyan")
            self._logger.log("info", f"{memory_type}: Agent: {item.agent}", color="cyan")
            self._logger.log("info", f"{memory_type}: Quality: {item.metadata.get('quality')}", color="cyan")
            
        metadata = item.metadata
        metadata.update({"agent": item.agent, "expected_output": item.expected_output})
        self.storage.save(  # type: ignore # BUG?: Unexpected keyword argument "task_description","score","datetime" for "save" of "Storage"
            task_description=item.task,
            score=metadata["quality"],
            metadata=metadata,
            datetime=item.datetime,
        )

    def search(self, task: str, latest_n: int = 3) -> List[Dict[str, Any]]:  # type: ignore # signature of "search" incompatible with supertype "Memory"
        return self.storage.load(task, latest_n)  # type: ignore # BUG?: "Storage" has no attribute "load"

    def reset(self) -> None:
        self.storage.reset()
