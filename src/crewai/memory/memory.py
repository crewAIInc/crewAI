from typing import Any, Dict, List, Optional

from crewai.memory.storage.rag_storage import RAGStorage
from crewai.utilities.logger import Logger


class Memory:
    """
    Base class for memory, now supporting agent tags and generic metadata.
    """

    def __init__(self, storage: RAGStorage, memory_verbose: bool = False):
        self.storage = storage
        self.memory_verbose = memory_verbose
        self._logger = Logger(verbose=memory_verbose)

    def save(
        self,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
        agent: Optional[str] = None,
    ) -> None:
        metadata = metadata or {}
        if agent:
            metadata["agent"] = agent
        
        if self.memory_verbose:
            memory_type = self.__class__.__name__
            agent_info = f" from agent '{agent}'" if agent else ""
            self._logger.log("info", f"{memory_type}: Saving{agent_info}: {value[:100]}{'...' if len(str(value)) > 100 else ''}", color="cyan")

        self.storage.save(value, metadata)

    def search(
        self,
        query: str,
        limit: int = 3,
        score_threshold: float = 0.35,
    ) -> List[Any]:
        if self.memory_verbose:
            memory_type = self.__class__.__name__
            self._logger.log("info", f"{memory_type}: Searching for: {query}", color="cyan")
            
        results = self.storage.search(
            query=query, limit=limit, score_threshold=score_threshold
        )
        
        if self.memory_verbose and results:
            self._logger.log("info", f"{memory_type}: Found {len(results)} results", color="cyan")
            
        return results
