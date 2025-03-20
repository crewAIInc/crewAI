import os
from typing import Any, Dict, List

from mem0 import MemoryClient

from crewai.memory.storage.interface import Storage


class Mem0Storage(Storage):
    """
    Extends Storage to handle embedding and searching across entities using Mem0.
    
    Supports configuring Redis as a vector store through the memory_config:
    
    ```python
    crew = Crew(
        memory=True,
        memory_config={
            "provider": "mem0",
            "config": {
                "user_id": "test-user",
                "api_key": "mem0-api-key",
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

        # Get configuration from memory_config
        config = self.memory_config.get("config", {})
        mem0_api_key = config.get("api_key") or os.getenv("MEM0_API_KEY")
        mem0_org_id = config.get("org_id")
        mem0_project_id = config.get("project_id")
        vector_store_config = config.get("vector_store")

        # If vector store configuration is provided, use Memory.from_config
        if vector_store_config:
            try:
                from mem0.memory.main import Memory
                
                # Prepare memory config with vector store configuration
                memory_config = {
                    "vector_store": vector_store_config
                }
                
                # Add API key if provided
                if mem0_api_key:
                    memory_config["api_key"] = mem0_api_key
                
                # Add org_id and project_id if provided
                if mem0_org_id:
                    memory_config["org_id"] = mem0_org_id
                if mem0_project_id:
                    memory_config["project_id"] = mem0_project_id
                    
                # Initialize Memory with configuration
                self.memory = Memory.from_config(memory_config)
            except ImportError:
                raise ImportError(
                    "Mem0 is not installed. Please install it with `pip install mem0ai`."
                )
            except Exception as e:
                raise ValueError(f"Failed to initialize Memory with vector store configuration: {e}")
        else:
            # Fall back to default MemoryClient initialization
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
