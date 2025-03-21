import os
from typing import Any, Dict, Optional

from crewai.memory.memory import Memory


class UserMemory(Memory):
    """
    UserMemory class for handling user memory storage and retrieval.
    Inherits from the Memory class and utilizes an instance of a class that
    adheres to the Storage for data storage, specifically working with
    MemoryItem instances.
    
    To configure with Redis as a vector store, provide a memory_config to the Crew:
    
    ```python
    crew = Crew(
        memory=True,
        memory_config={
            "provider": "mem0",
            "config": {
                "user_id": "your-user-id",
                "api_key": os.getenv("MEM0_API_KEY"),  # Use environment variable
                "vector_store": {
                    "provider": "redis",
                    "config": {
                        "collection_name": "collection_name",
                        "embedding_model_dims": 1536,
                        "redis_url": "redis://redis-host:6379/0"
                    }
                }
            }
        }
    )
    ```
    """

    def __init__(self, crew=None):
        try:
            from crewai.memory.storage.mem0_storage import Mem0Storage
        except ImportError:
            raise ImportError(
                "Mem0 is not installed. Please install it with `pip install mem0ai`."
            )
        storage = Mem0Storage(type="user", crew=crew)
        super().__init__(storage)

    def save(
        self,
        value,
        metadata: Optional[Dict[str, Any]] = None,
        agent: Optional[str] = None,
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
        results = self.storage.search(
            query=query,
            limit=limit,
            score_threshold=score_threshold,
        )
        return results
