from typing import Any, Dict, List, Optional

from crewai.memory.storage.rag_storage import RAGStorage


class Memory:
    """
    Base class for memory, now supporting agent tags, generic metadata, and custom keys.
    
    Custom keys allow scoping memories to specific entities (users, accounts, sessions),
    retrieving memories contextually, and preventing data leakage across logical boundaries.
    """

    def __init__(self, storage: RAGStorage):
        self.storage = storage

    def save(
        self,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
        agent: Optional[str] = None,
        custom_key: Optional[str] = None,
    ) -> None:
        metadata = metadata or {}
        if agent:
            metadata["agent"] = agent
        if custom_key:
            metadata["custom_key"] = custom_key

        self.storage.save(value, metadata)

    def search(
        self,
        query: str,
        limit: int = 3,
        score_threshold: float = 0.35,
        custom_key: Optional[str] = None,
    ) -> List[Any]:
        filter_dict = None
        if custom_key:
            filter_dict = {"custom_key": {"$eq": custom_key}}
            
        return self.storage.search(
            query=query, limit=limit, score_threshold=score_threshold, filter=filter_dict
        )
