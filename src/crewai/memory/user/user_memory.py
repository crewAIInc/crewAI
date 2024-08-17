from crewai.memory.memory import Memory
from crewai.memory.user.user_memory_item import UserMemoryItem
from crewai.memory.storage.mem0_storage import Mem0Storage


class UserMemory(Memory):
    """
    UserMemory class for handling user memory storage and retrieval.
    Inherits from the Memory class and utilizes an instance of a class that
    adheres to the Storage for data storage, specifically working with
    MemoryItem instances.
    """

    def __init__(self, crew=None):
        storage = Mem0Storage(type="user", crew=crew)
        super().__init__(storage)

    def save(self, item: UserMemoryItem) -> None:
        data = f"Remember the details about the user: {item.data}"
        super().save(data, item.metadata, user=item.user)

    def search(self, query: str, score_threshold: float = 0.35):
        return self.storage.search(query=query, score_threshold=score_threshold)
