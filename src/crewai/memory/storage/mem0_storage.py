import os
from typing import Any, Dict, List

from mem0 import Memory, MemoryClient

from crewai.memory.storage.interface import Storage

MAX_AGENT_ID_LENGTH_MEM0 = 255


class Mem0Storage(Storage):
    """
    Extends Storage to handle embedding and searching across entities using Mem0.
    """
    def __init__(self, type, crew=None, config=None):
        super().__init__()

        self._validate_type(type)
        self.memory_type = type
        self.crew = crew

        # TODO: Memory config will be removed in the future the config will be passed as a parameter
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
        self.infer = cfg.get("infer", False)

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

    def _create_filter_for_search(self):
        """
        Returns:
            dict: A filter dictionary containing AND conditions for querying data.
                - Includes user_id if memory_type is 'external'.
                - Includes run_id if memory_type is 'short_term' and mem0_run_id is present.
        """
        filter = {
            "AND": []
        }

        # Add user_id condition if the memory type is external
        if self.memory_type == "external":
            filter["AND"].append({"user_id": self.config.get("user_id", "")})

        # Add run_id condition if the memory type is short_term and a run ID is set
        if self.memory_type == "short_term" and self.mem0_run_id:
            filter["AND"].append({"run_id": self.mem0_run_id})

        return filter

    def save(self, value: Any, metadata: Dict[str, Any]) -> None:
        user_id = self.config.get("user_id", "")
        assistant_message = [{"role" : "assistant","content" : value}] 

        base_metadata = {
            "short_term": "short_term",
            "long_term": "long_term",
            "entities": "entity",
            "external": "external"
        }

        # Shared base params
        params: dict[str, Any] = {
            "metadata": {"type": base_metadata[self.memory_type], **metadata},
            "infer": self.infer
        }

        if self.memory_type == "external":
            params["user_id"] = user_id

        
        if params:
            # MemoryClient-specific overrides
            if isinstance(self.memory, MemoryClient):
                params["includes"] = self.includes
                params["excludes"] = self.excludes
                params["output_format"] = "v1.1"
                params["version"]="v2"

                if self.memory_type == "short_term":
                    params["run_id"] = self.mem0_run_id

            self.memory.add(assistant_message, **params)

    def search(self,query: str,limit: int = 3,score_threshold: float = 0.35) -> List[Any]:
        params = {
            "query": query, 
            "limit": limit, 
            "version": "v2",
            "output_format": "v1.1"
            }
        
        if user_id := self.config.get("user_id", ""):
            params["user_id"] = user_id

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

        params["filters"] = self._create_filter_for_search()
        params['threshold'] = score_threshold

        if isinstance(self.memory, Memory):
            del params["metadata"], params["version"], params["run_id"], params['output_format']

        results = self.memory.search(**params)
        return [r for r in results["results"]]
    
    def reset(self):
        if self.memory:
            self.memory.reset()
