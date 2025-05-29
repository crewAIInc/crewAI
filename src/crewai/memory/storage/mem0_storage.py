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

        self._validate_type(type)
        self.memory_type = type
        self.crew = crew
        self.config = config or getattr(crew, "memory_config", {}).get("config", {}) or {}

        self._validate_user_id()
        self._extract_config_values()
        self._initialize_memory()

    def _validate_type(self, type):
        supported_types = {"user", "short_term", "long_term", "entities", "external"}
        if type not in supported_types:
            raise ValueError(
                f"Invalid type '{type}' for Mem0Storage. Must be one of: {', '.join(supported_types)}"
            )

    def _validate_user_id(self):
        if self.memory_type == "user" and not self.config.get("user_id", ""):
            raise ValueError("User ID is required for user memory type")

    def _extract_config_values(self):
        cfg = self.config
        self.mem0_run_id = cfg.get("run_id")
        self.includes = cfg.get("includes")
        self.excludes = cfg.get("excludes")
        self.custom_categories = cfg.get("custom_categories")

    def _initialize_memory(self):
        api_key = self.config.get("api_key") or os.getenv("MEM0_API_KEY")
        org_id = self.config.get("org_id")
        project_id = self.config.get("project_id")
        local_config = self.config.get("local_mem0_config")

        if api_key:
            self.memory = (
                MemoryClient(api_key=api_key, org_id=org_id, project_id=project_id)
                if org_id and project_id
                else MemoryClient(api_key=api_key)
            )
            if self.custom_categories:
                self.memory.update_project(custom_categories=self.custom_categories)
        else:
            self.memory = (
                Memory.from_config(local_config)
                if local_config and len(local_config)
                else Memory()
            )

    def _get_agent_name(self) -> str:
        if not self.crew:
            return ""

        agents = self.crew.agents
        agents = [self._sanitize_role(agent.role) for agent in agents]
        agents = "_".join(agents)
        return agents

    def _sanitize_role(self, role: str) -> str:
        """
        Sanitizes agent roles to ensure valid directory names.
        """
        return role.replace("\n", "").replace(" ", "_").replace("/", "_")

    def save(self, value: Any, metadata: Dict[str, Any]) -> None:
        user_id = self.config.get("user_id", "")
        agent_name = self._get_agent_name()

        base_metadata = {
            "short_term": "short_term",
            "long_term": "long_term",
            "entities": "entity",
            "external": "external"
        }

        # Shared base params
        params: dict[str, Any] = {
            "agent_id": agent_name,
            "metadata": {"type": base_metadata[self.memory_type], **metadata}
        }

        # Type-specific overrides
        if self.memory_type == "short_term":
            params["infer"] = False
        elif self.memory_type == "long_term":
            params["infer"] = False
        elif self.memory_type == "entities":
            params["infer"] = False
        elif self.memory_type == "external":
            params["user_id"] = user_id

        
        if params:
            # MemoryClient-specific overrides
            if isinstance(self.memory, MemoryClient):
                params["version"] = "v2"
                params["includes"] = self.includes
                params["excludes"] = self.excludes

                if self.memory_type == "short_term":
                    params["run_id"] = self.mem0_run_id

            self.memory.add(value, **params)

    def search(self,query: str,limit: int = 3,score_threshold: float = 0.35) -> List[Any]:
        params = {
            "query": query, 
            "limit": limit, 
            "version": "v2"
            }
        
        if user_id := self.config.get("user_id", ""):
            params["user_id"] = user_id

        agent_name = self._get_agent_name()
        params["agent_id"] = agent_name

        memory_type_map = {
            "short_term": {"type": "short_term"},
            "long_term": {"type": "long_term"},
            "entities": {"type": "entity"},
            "external": {"type": "external"},
        }
        
        if self.memory_type in memory_type_map:
            params["metadata"] = memory_type_map[self.memory_type]
            if self.memory_type == "short_term":
                params["run_id"] = self.mem0_run_id

        # Discard the filters for now since we create the filters
        # automatically when the crew is created.
        if isinstance(self.memory, Memory):
            del params["metadata"], params["version"], params["run_id"]
            
        results = self.memory.search(**params)
        return [r for r in results["results"] if r["score"] >= score_threshold]

    def reset(self):
        if self.memory:
            self.memory.reset()
