import warnings
from typing import Any

from crewai.memory.memory import Memory


class UserMemory(Memory):
    """UserMemory class for handling user memory storage and retrieval.
    Inherits from the Memory class and utilizes an instance of a class that
    adheres to the Storage for data storage, specifically working with
    MemoryItem instances.
    """

    def __init__(self, crew=None) -> None:
        warnings.warn(
            "UserMemory is deprecated and will be removed in a future version. "
            "Please use ExternalMemory instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            from crewai.memory.storage.mem0_storage import Mem0Storage
        except ImportError:
            msg = "Mem0 is not installed. Please install it with `pip install mem0ai`."
            raise ImportError(
                msg,
            )
        storage = Mem0Storage(type="user", crew=crew)
        super().__init__(storage)

    def save(
        self,
        value,
        metadata: dict[str, Any] | None = None,
        agent: str | None = None,
    ) -> None:
        # TODO: Change this function since we want to take care of the case where we save memories for the usr
        data = f"Remember the details about the user: {value}"
        super().save(data, metadata)

    def search(
        self,
        query: str,
        limit: int = 3,
        score_threshold: float = 0.35,
    ):
        return self.storage.search(
            query=query,
            limit=limit,
            score_threshold=score_threshold,
        )

    def reset(self) -> None:
        try:
            self.storage.reset()
        except Exception as e:
            msg = f"An error occurred while resetting the user memory: {e}"
            raise Exception(msg)
