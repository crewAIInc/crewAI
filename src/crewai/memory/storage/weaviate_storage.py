import os
from typing import Any, Dict, List

import weaviate
from weaviate.classes.init import Auth

from crewai.memory.storage.interface import Storage
from crewai.utilities.chromadb import sanitize_collection_name

class WeaviateStorage(Storage):
    """
    Extends Storage to handle storing and searching across memories in Weaviate.
    """

    def __init__(self, type, crew=None, config=None):
        super().__init__()
        supported_types = ["user", "short_term", "long_term", "entities", "external"]
        if type not in supported_types:
            raise ValueError(
                f"Invalid type '{type}' for WeaviateStorage. Must be one of: "
                + ", ".join(supported_types)
            )

        self.memory_type = type
        self.crew = crew
        self.config = config or {}
        # TODO: Memory config will be removed in the future the config will be passed as a parameter
        self.memory_config = self.config or getattr(crew, "memory_config", {}) or {}

        # User ID is required for user memory type "user" since it's used as a unique identifier for the user.
        user_id = self._get_user_id()
        if type == "user" and not user_id:
            raise ValueError("User ID is required for user memory type")

        # API key in memory config overrides the environment variable
        config = self._get_config()
        weaviate_api_key = config.get("api_key") or os.getenv("WEAVIATE_API_KEY")
        weaviate_url = config.get("url") or os.getenv("WEAVIATE_URL")
        local_config = config.get("local_weaviate_config")

        if weaviate_api_key and weaviate_url:
            # Connect to a remote Weaviate instance (e.g., Weaviate Cloud)
            self.client = weaviate.connect_to_weaviate_cloud(
                cluster_url=weaviate_url,
                auth_credentials=Auth.api_key(weaviate_api_key)
            )
        elif local_config:
            # Custom local configuration (e.g., custom ports or hosts)
            self.client = weaviate.connect_to_custom(**local_config)
        else:
            # Default to local instance
            self.client = weaviate.connect_to_local()

    def _sanitize_role(self, role: str) -> str:
        """
        Sanitizes agent roles to ensure valid directory names.
        """
        return role.replace("\n", "").replace(" ", "_").replace("/", "_")

    def save(self, value: Any, metadata: Dict[str, Any]) -> None:
        user_id = self._get_user_id()
        agent_name = self._get_agent_name()
        params = None
        if self.memory_type == "short_term":
            params = {
                "agent_id": agent_name,
                "infer": False,
                "metadata": {"type": "short_term", **metadata},
            }
        elif self.memory_type == "long_term":
            params = {
                "agent_id": agent_name,
                "infer": False,
                "metadata": {"type": "long_term", **metadata},
            }
        elif self.memory_type == "entities":
            params = {
                "agent_id": agent_name,
                "infer": False,
                "metadata": {"type": "entity", **metadata},
            }
        elif self.memory_type == "external":
            params = {
                "user_id": user_id,
                "agent_id": agent_name,
                "metadata": {"type": "external", **metadata},
            }

        if params:
            # UPDATE ME!
            if isinstance(self.memory, weaviate.WeaviateClient):
                params["output_format"] = "v4.15.4"
            self.memory.add(value, **params)

    def search(
        self,
        query: str,
        limit: int = 3,
    ) -> List[Any]:
        params = {"query": query, "limit": limit, "output_format": "v1.1"}
        if user_id := self._get_user_id():
            params["user_id"] = user_id

        agent_name = self._get_agent_name()
        if self.memory_type == "short_term":
            params["agent_id"] = agent_name
            params["metadata"] = {"type": "short_term"}
        elif self.memory_type == "long_term":
            params["agent_id"] = agent_name
            params["metadata"] = {"type": "long_term"}
        elif self.memory_type == "entities":
            params["agent_id"] = agent_name
            params["metadata"] = {"type": "entity"}
        elif self.memory_type == "external":
            params["agent_id"] = agent_name
            params["metadata"] = {"type": "external"}
        # Check data model for this
        pass

    def _get_user_id(self) -> str:
        return self._get_config().get("user_id", "")

    def _get_agent_name(self) -> str:
        if not self.crew:
            return ""

        agents = self.crew.agents
        agents = [self._sanitize_role(agent.role) for agent in agents]
        agents = "_".join(agents)
        return sanitize_collection_name(name=agents,max_collection_length=255) # Check this

    def _get_config(self) -> Dict[str, Any]:
        return self.config or getattr(self, "memory_config", {}).get("config", {}) or {}

    def reset(self):
        if self.memory:
            self.memory.reset()
