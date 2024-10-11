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

    def __init__(self, storage=None):
        storage = storage if storage else LTMSQLiteStorage()
        super().__init__(storage)

    def save(self, item: LongTermMemoryItem) -> None:
        metadata = item.metadata.copy()  # Create a copy to avoid modifying the original
        metadata.update(
            {
                "agent": item.agent,
                "expected_output": item.expected_output,
                "quality": item.quality,  # Add quality to metadata
            }
        )
        self.storage.save(  # type: ignore # BUG?: Unexpected keyword argument "task_description","score","datetime" for "save" of "Storage"
            task_description=item.task,
            score=item.quality,
            metadata=metadata,
            datetime=item.datetime,
        )

    def search(self, task: str, latest_n: int = 3) -> List[Dict[str, Any]]:
        results = self.storage.load(task, latest_n)
        return results

    def reset(self) -> None:
        self.storage.reset()
