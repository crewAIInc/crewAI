import logging
import os
import uuid
from typing import Any, Dict, List, Optional

import weaviate
import weaviate.classes as wvc
from weaviate.auth import AuthApiKey

from crewai.rag.storage.base_rag_storage import BaseRAGStorage


class WeaviateStorage(BaseRAGStorage):
    """
    Extends Storage to handle embeddings for memory entries using Weaviate.
    """

    client: weaviate.Client | None = None
    collection = None

    def __init__(
        self, type, allow_reset=True, embedder_config=None, crew=None, path=None
    ):
        super().__init__(type, allow_reset, embedder_config, crew)
        
        # Roles
        self.raw_roles: list[str] = self._get_agent_roles(crew)
        self.sanitized_roles: list[str] = [self._sanitize_role(r) for r in self.raw_roles]

        self.type = type
        self.allow_reset = allow_reset

        self._initialize_app()

    def _set_embedder_config(self):
        """Configure embedder settings"""
        pass

    def _initialize_app(self):
        """Initialize Weaviate client and create/get collection"""
        try:
            # Connect to Weaviate Client
            self.client = weaviate.connect_to_weaviate_cloud(
                cluster_url=os.getenv("WEAVIATE_URL"),
                auth_credentials=AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
            )
            
            # Create or get collection
            self._initialize_collection()
            
            logging.info(f"Weaviate collection initialized: {self.collection_name}")
            
        except Exception as e:
            logging.error(f"Error initializing Weaviate client: {str(e)}")
            raise

    def _initialize_collection(self):
        """Create collection if it doesn't exist, otherwise get existing"""
        self.collection_name = "AgentMemories"
        
        if not self.client.collections.exists(self.collection_name):
            self.collection = self.client.collections.create(
                name=self.collection_name, # Should we let this be configurable?
                vectorizer_config=self._get_vectorizer_config(),
                multi_tenancy_config = wvc.config.Configure.multi_tenancy(
                    enabled=True,
                    auto_tenant_creation=True,
                    auto_tenant_activation=True,
                ),
                properties=[
                    wvc.config.Property(
                        name="output",
                        data_type=wvc.config.DataType.TEXT,
                        description="The output sent from the agent."
                    ),
                    wvc.config.Property(
                        name="messages",
                        data_type=wvc.config.DataType.TEXT,
                        description="The messages used as input for this agent inference."
                    ),
                    wvc.config.Property(
                        name="task_description",
                        data_type=wvc.config.DataType.TEXT,
                        description="The description of the task that the agent is working on."
                    ),
                    wvc.config.Property(
                        name="timestamp",
                        data_type=wvc.config.DataType.NUMBER,
                        description="Unix timestamp of when memory was created"
                    ),
                ],
            )
            logging.info(f"Created new Weaviate collection: {self.collection_name}")
        else:
            # Get existing collection
            self.collection = self.client.collections.get(self.collection_name)
            logging.info(f"Using existing Weaviate collection: {self.collection_name}")
            
        # Ensure tenants exist for each agent role
        if self.sanitized_roles:
            try:
                self.collection.tenants.create(
                    tenants=self.sanitized_roles
                )
                logging.info(f"Ensured tenants for roles: {self.sanitized_roles}")
            except Exception as e:
                logging.debug(f"Tenant creation note: {e}")

    def _get_vectorizer_config(self):
        """Get appropriate vectorizer config based on embedder settings"""
        # Default to Weaviate's text2vec module
        # You can extend this to support other vectorizers based on embedder_config
        return wvc.config.Configure.Vectorizer.text2vec_weaviate()

    def _get_agent_roles(self, crew) -> list[str]:
        """Return raw role strings in the same order the crew defines them."""
        if not crew or not getattr(crew, "agents", None):
            return []
        roles = []
        for agent in crew.agents:
            role = getattr(agent, "role", None)
            if isinstance(role, str) and role.strip():
                roles.append(role.strip())
        return roles

    def _sanitize_role(self, role: str) -> str:
        """Sanitize role to a valid tenant name."""
        sanitized = role.replace("\n", "").replace(" ", "_").replace("/", "_")
        sanitized = ''.join(c for c in sanitized if c.isalnum() or c == '_')
        if sanitized and not sanitized[0].isalpha():
            sanitized = 'M' + sanitized
        return sanitized

    # update to select the tenant with `agent_role`
    def save(self, value: Any, metadata: Dict[str, Any]) -> None:
        """Save a memory entry to Weaviate"""
        if self.collection is None:
            self._initialize_app()

        # check if the role is already a tenant in the collection
        
        try:
            # Prepare data object
            import json
            import time

            raw_role = metadata.get("agent", "unknown")
            sanitized_role = self._sanitize_role(raw_role)

            agent_tenant = self.collection.with_tenant(sanitized_role)
            
            data_object = {
                "output": str(value),
                "messages": json.dumps(metadata.get("messages", [])),
                "task_description": metadata.get("description", ""),
                "timestamp": time.time()
            }
            
            agent_tenant.data.insert(
                properties=data_object,
                uuid=str(uuid.uuid4())
            )
                
        except Exception as e:
            logging.error(f"Error during {self.type} save to Weaviate: {str(e)}")

    def search(
        self,
        query: str,
        limit: int = 3,
        score_threshold: float = 0.35,
        agent_role: Optional[str] = None,
    ) -> List[Any]:
        """Search for relevant memories using Weaviate's hybrid search"""
        if self.collection is None:
            self._initialize_app()

        try:
            agent_tenant = self.collection.with_tenant(agent_role)
            response = agent_tenant.query.hybrid(
                query=query,
                limit=limit,
            )
            
            # Parse and format results
            results = []
            for item in response.objects:
                if item["score"] >= score_threshold:
                    results.append({
                        "id": str(item.uuid),
                        "metadata_json": item.properties["metadata_json"],
                        "context": item.properties["content"]
                    })
            
            return results
            
        except Exception as e:
            logging.error(f"Error during {self.type} search in Weaviate: {str(e)}")
            return []

    def reset(self) -> None:
        """Reset the storage by deleting the collection"""
        try:
            if self.client and self.collection_name:
                if self.client.collections.exists(self.collection_name):
                    self.client.collections.delete(self.collection_name)
                    logging.info(f"Deleted Weaviate collection: {self.collection_name}")
                
                self.collection = None
                
        except Exception as e:
            logging.error(f"Error resetting Weaviate storage: {str(e)}")
            raise Exception(
                f"An error occurred while resetting the {self.type} memory: {e}"
            )

    def close(self):
        """Cleanup Weaviate client connection"""
        self.client.close()
    
    def _generate_embedding(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Generate an embedding for the given text and metadata."""
        pass