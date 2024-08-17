import os
from typing import Any, Dict, List, Optional

from mem0 import MemoryClient
from crewai.memory.storage.interface import Storage


class Mem0Storage(Storage):
    """
    Extends Storage to handle embedding and searching across entities using Mem0.
    """

    def __init__(self, type, crew=None):
        super().__init__()
        if (
            not os.getenv("OPENAI_API_KEY")
            and not os.getenv("OPENAI_BASE_URL") == "https://api.openai.com/v1"
        ):
            os.environ["OPENAI_API_KEY"] = "fake"

        agents = crew.agents if crew else []
        agents = [agent.role for agent in agents]
        agents = "_".join(agents)

        self.app_id = agents
        self.memory = MemoryClient(api_key=os.getenv("MEM0_API_KEY"))

    def save(self, value: Any, metadata: Dict[str, Any]) -> None:
        self.memory.add(value, metadata=metadata, app_id=self.app_id)

    def search(
        self,
        query: str,
        limit: int = 3,
        filters: Optional[dict] = None,
        score_threshold: float = 0.35,
    ) -> List[Any]:
        params = {"query": query, "limit": limit, "app_id": self.app_id}
        if filters:
            params["filters"] = filters
        results = self.memory.search(**params)
        return [r for r in results if r["score"] >= score_threshold]
