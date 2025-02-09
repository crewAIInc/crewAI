import os
from typing import Any, Dict, List

from mem0 import MemoryClient

from crewai.memory.storage.interface import Storage


class Mem0Storage(Storage):
    """
    Extends Storage to handle embedding and searching across entities using Mem0.
    """

    def __init__(self, type, crew=None):
        super().__init__()

        if type not in ["user", "short_term", "long_term", "entities"]:
            raise ValueError("Invalid type for Mem0Storage. Must be 'user' or 'agent'.")

        self.memory_type = type
        self.crew = crew
        self.memory_config = crew.memory_config

        # User ID is required for user memory type "user" since it's used as a unique identifier for the user.
        user_id = self._get_user_id()
        if type == "user" and not user_id:
            raise ValueError("User ID is required for user memory type")

        # API key in memory config overrides the environment variable
        config = self.memory_config.get("config", {})
        mem0_api_key = config.get("api_key") or os.getenv("MEM0_API_KEY")
        mem0_org_id = config.get("org_id")
        mem0_project_id = config.get("project_id")

        # Initialize MemoryClient with available parameters
        if mem0_org_id and mem0_project_id:
            self.memory = MemoryClient(
                api_key=mem0_api_key, org_id=mem0_org_id, project_id=mem0_project_id
            )
        else:
            self.memory = MemoryClient(api_key=mem0_api_key)

    def _sanitize_role(self, role: str) -> str:
        """
        Sanitizes agent roles to ensure valid directory names.
        """
        return role.replace("\n", "").replace(" ", "_").replace("/", "_")

    def save(self, value: Any, metadata: Dict[str, Any]) -> None:
        user_id = self._get_user_id()
        agent_name = self._get_agent_name()
        if self.memory_type == "user":
            self.memory.add(value, user_id=user_id, metadata={**metadata})
        elif self.memory_type == "short_term":
            agent_name = self._get_agent_name()
            self.memory.add(
                value, agent_id=agent_name, metadata={"type": "short_term", **metadata}
            )
        elif self.memory_type == "long_term":
            agent_name = self._get_agent_name()
            self.memory.add(
                value,
                agent_id=agent_name,
                infer=False,
                metadata={"type": "long_term", **metadata},
            )
        elif self.memory_type == "entities":
            entity_name = self._get_agent_name()
            self.memory.add(
                value, user_id=entity_name, metadata={"type": "entity", **metadata}
            )

    def search(
        self,
        query: str,
        limit: int = 3,
        score_threshold: float = 0.35,
    ) -> List[Any]:
        params = {"query": query, "limit": limit}
        if self.memory_type == "user":
            user_id = self._get_user_id()
            params["user_id"] = user_id
        elif self.memory_type == "short_term":
            agent_name = self._get_agent_name()
            params["agent_id"] = agent_name
            params["metadata"] = {"type": "short_term"}
        elif self.memory_type == "long_term":
            agent_name = self._get_agent_name()
            params["agent_id"] = agent_name
            params["metadata"] = {"type": "long_term"}
        elif self.memory_type == "entities":
            agent_name = self._get_agent_name()
            params["agent_id"] = agent_name
            params["metadata"] = {"type": "entity"}

        # Discard the filters for now since we create the filters
        # automatically when the crew is created.
        results = self.memory.search(**params)
        return [r for r in results if r["score"] >= score_threshold]

    def _get_user_id(self):
        if self.memory_type == "user":
            if hasattr(self, "memory_config") and self.memory_config is not None:
                return self.memory_config.get("config", {}).get("user_id")
            else:
                return None
        return None

    def _get_agent_name(self):
        agents = self.crew.agents if self.crew else []
        agents = [self._sanitize_role(agent.role) for agent in agents]
        agents = "_".join(agents)
        return agents
