import os
from typing import Any, Dict, List

from mem0 import Memory, MemoryClient

from crewai.memory.storage.interface import Storage


class Mem0Storage(Storage):
    """
    Extends Storage to handle embedding and searching across entities using Mem0.
    
    Supports Mem0 v2 API with run_id for associating memories with specific conversation
    sessions. By default, uses v2 API which is recommended for better context management.
    
    Args:
        type: The type of memory storage ("user", "short_term", "long_term", "entities", "external")
        crew: The crew instance this storage is associated with
        config: Optional configuration dictionary that overrides crew.memory_config
        
    Configuration options:
        version: API version to use ("v1.1" or "v2", defaults to "v2")
        run_id: Optional session identifier for associating memories with specific conversations
        api_key: Mem0 API key (defaults to MEM0_API_KEY environment variable)
        user_id: User identifier (required for "user" memory type)
        org_id: Optional organization ID for Mem0 API
        project_id: Optional project ID for Mem0 API
        local_mem0_config: Optional configuration for local Mem0 instance
    """

    SUPPORTED_VERSIONS = ["v1.1", "v2"]
    
    DEFAULT_VERSION = "v2"

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
        
        config = self._get_config()
        self.version = config.get("version", self.DEFAULT_VERSION)
        self.run_id = config.get("run_id")
        
        self._validate_config()

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

    def _validate_config(self) -> None:
        """
        Validate configuration parameters.
        
        Raises:
            ValueError: If the version is not supported
        """
        if self.version not in self.SUPPORTED_VERSIONS:
            raise ValueError(
                f"Unsupported version: {self.version}. "
                f"Please use one of: {', '.join(self.SUPPORTED_VERSIONS)}"
            )
        
        if self.run_id is not None and not isinstance(self.run_id, str):
            raise ValueError("run_id must be a string")

    def _build_params(self, base_params: Dict[str, Any], method: str = "add") -> Dict[str, Any]:
        """
        Centralize parameter building for API calls.
        
        Args:
            base_params: Base parameters to build upon
            method: The method being called ("add" or "search")
            
        Returns:
            Dict[str, Any]: Complete parameters for API call
        """
        params = base_params.copy()
        
        # Add version and run_id for MemoryClient
        if isinstance(self.memory, MemoryClient):
            params["version"] = self.version
            
            if self.run_id:
                params["run_id"] = self.run_id
        elif isinstance(self.memory, Memory) and method == "search" and "metadata" in params:
            del params["metadata"]
            
        return params

    def _sanitize_role(self, role: str) -> str:
        """
        Sanitizes agent roles to ensure valid directory names.
        
        Args:
            role: The role name to sanitize
            
        Returns:
            str: Sanitized role name
        """
        return role.replace("\n", "").replace(" ", "_").replace("/", "_")

    def save(self, value: Any, metadata: Dict[str, Any]) -> None:
        """
        Save a memory item.
        
        Args:
            value: The memory content to save
            metadata: Additional metadata for the memory
        """
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
            params = self._build_params(params, method="add")
            self.memory.add(value, **params)

    def search(
        self,
        query: str,
        limit: int = 3,
        score_threshold: float = 0.35,
    ) -> List[Any]:
        """
        Search for memories.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            score_threshold: Minimum score for results to be included
            
        Returns:
            List[Any]: List of memory items that match the query
        """
        base_params = {"query": query, "limit": limit}
        if user_id := self._get_user_id():
            base_params["user_id"] = user_id

        agent_name = self._get_agent_name()
        if self.memory_type == "short_term":
            base_params["agent_id"] = agent_name
            base_params["metadata"] = {"type": "short_term"}
        elif self.memory_type == "long_term":
            base_params["agent_id"] = agent_name
            base_params["metadata"] = {"type": "long_term"}
        elif self.memory_type == "entities":
            base_params["agent_id"] = agent_name
            base_params["metadata"] = {"type": "entity"}
        elif self.memory_type == "external":
            base_params["agent_id"] = agent_name
            base_params["metadata"] = {"type": "external"}

        params = self._build_params(base_params, method="search")
        results = self.memory.search(**params)
        
        if isinstance(results, dict) and "results" in results:
            return [r for r in results["results"] if r["score"] >= score_threshold]
        elif isinstance(results, list):
            return [r for r in results if r["score"] >= score_threshold]
        else:
            return []

    def _get_user_id(self) -> str:
        """
        Get the user ID from configuration.
        
        Returns:
            str: User ID or empty string if not found
        """
        return self._get_config().get("user_id", "")

    def _get_agent_name(self) -> str:
        """
        Get the agent name from the crew.
        
        Returns:
            str: Agent name or empty string if not found
        """
        if not self.crew:
            return ""

        agents = self.crew.agents
        agents = [self._sanitize_role(agent.role) for agent in agents]
        agents = "_".join(agents)
        return agents

    def _get_config(self) -> Dict[str, Any]:
        """
        Get the configuration from either config or memory_config.
        
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        return self.config or getattr(self, "memory_config", {}).get("config", {}) or {}

    def reset(self) -> None:
        """
        Reset the memory.
        """
        if self.memory:
            self.memory.reset()
