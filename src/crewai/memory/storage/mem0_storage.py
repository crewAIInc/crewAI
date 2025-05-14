import os
from typing import Any, Dict, List

from mem0 import Memory, MemoryClient

from crewai.memory.storage.interface import Storage


class Mem0Storage(Storage):
    """
    Extends Storage to handle embedding and searching across entities using Mem0.
    """

    def __init__(self, type, crew=None, config=None):
        super().__init__()
        supported_types = ["user", "short_term", "long_term", "entities", "external"]
        if type not in supported_types:
            raise ValueError(
                f"Invalid type '{type}' for Mem0Storage. Must be one of: "
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
        mem0_api_key = config.get("api_key") or os.getenv("MEM0_API_KEY")
        mem0_org_id = config.get("org_id")
        mem0_project_id = config.get("project_id")
        mem0_local_config = config.get("local_mem0_config")

        # Initialize MemoryClient or Memory based on the presence of the mem0_api_key
        if mem0_api_key:
            if mem0_org_id and mem0_project_id:
                self.memory = MemoryClient(
                    api_key=mem0_api_key, org_id=mem0_org_id, project_id=mem0_project_id
                )
            else:
                self.memory = MemoryClient(api_key=mem0_api_key)
        else:
            if mem0_local_config and len(mem0_local_config):
                self.memory = Memory.from_config(mem0_local_config)
            else:
                self.memory = Memory()

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
            if isinstance(self.memory, MemoryClient):
                params["output_format"] = "v1.1"
            self.memory.add(value, **params)

    def search(
        self,
        query: str,
        limit: int = 3,
        score_threshold: float = 0.35,
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

        # Discard the filters for now since we create the filters
        # automatically when the crew is created.
        if isinstance(self.memory, Memory):
            del params["metadata"], params["output_format"]
            
        results = self.memory.search(**params)
        return [r for r in results["results"] if r["score"] >= score_threshold]

    def _get_user_id(self) -> str:
        return self._get_config().get("user_id", "")

    def _get_agent_name(self) -> str:
        if not self.crew:
            return ""

        agents = self.crew.agents
        agents = [self._sanitize_role(agent.role) for agent in agents]
        agents = "_".join(agents)
        return agents

    def _get_config(self) -> Dict[str, Any]:
        return self.config or getattr(self, "memory_config", {}).get("config", {}) or {}

    def reset(self):
        if self.memory:
            self.memory.reset()
