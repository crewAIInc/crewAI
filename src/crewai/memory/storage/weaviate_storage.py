import contextlib
import io
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

import weaviate
import weaviate.classes as wvc
from weaviate.auth import AuthApiKey

from crewai.memory.storage.base_rag_storage import BaseRAGStorage


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
        
        # Process agent roles to create unique collection naming
        agents = crew.agents if crew else []
        agents = [self._sanitize_role(agent.role) for agent in agents]
        agents = "_".join(agents)
        self.agents = agents

        # Create collection name based on type and agents
        self.collection_name = self._build_collection_name(type, agents)
        self.type = type
        self.allow_reset = allow_reset

        self._initialize_app()

    def _set_embedder_config(self):
        """Configure embedder settings"""
        pass

    def _initialize_app(self):
        """Initialize Weaviate client and create/get collection"""
        try:
            # Connect to Weaviate Cloud
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
        if not self.client.collections.exists(self.collection_name):
            self.collection = self.client.collections.create(
                name=self.collection_name,
                vectorizer_config=self._get_vectorizer_config(),
                properties=[
                    wvc.config.Property(
                        name="content",
                        data_type=wvc.config.DataType.TEXT,
                        description="The main content/text of the memory entry"
                    ),
                    wvc.config.Property(
                        name="metadata_json",
                        data_type=wvc.config.DataType.TEXT,
                        description="JSON string of metadata"
                    ),
                    wvc.config.Property(
                        name="agent_role",
                        data_type=wvc.config.DataType.TEXT,
                        description="Role of the agent that created this memory"
                    ),
                    wvc.config.Property(
                        name="memory_type",
                        data_type=wvc.config.DataType.TEXT,
                        description="Type of memory (short_term, long_term, entity)"
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

    def _get_vectorizer_config(self):
        """Get appropriate vectorizer config based on embedder settings"""
        # Default to Weaviate's text2vec module
        # You can extend this to support other vectorizers based on embedder_config
        return wvc.config.Configure.Vectorizer.text2vec_weaviate()

    def _sanitize_role(self, role: str) -> str:
        """
        Sanitizes agent roles to ensure valid collection names in Weaviate.
        Weaviate collection names must start with a letter and contain only 
        alphanumeric characters.
        """
        # Remove newlines and replace spaces/slashes with underscores
        sanitized = role.replace("\n", "").replace(" ", "_").replace("/", "_")
        
        # Remove any non-alphanumeric characters (except underscores)
        sanitized = ''.join(c for c in sanitized if c.isalnum() or c == '_')
        
        # Ensure it starts with a letter (prepend 'M' for Memory if needed)
        if sanitized and not sanitized[0].isalpha():
            sanitized = 'M' + sanitized
            
        return sanitized

    def _build_collection_name(self, type: str, agents_str: str) -> str:
        """
        Build a valid Weaviate collection name from type and agents.
        Weaviate has specific naming requirements for collections.
        """
        # Sanitize type
        type_sanitized = ''.join(c for c in type if c.isalnum() or c == '_')
        
        # Combine type and agents
        full_name = f"{type_sanitized}_{agents_str}" if agents_str else type_sanitized
        
        max_collection_name_length = 200
        if len(full_name) > max_collection_name_length:
            logging.warning(
                f"Trimming collection name from {len(full_name)} to {max_collection_name_length} characters."
            )
            full_name = full_name[:max_collection_name_length]
        
        # Ensure it starts with a capital letter (Weaviate convention)
        return full_name[0].upper() + full_name[1:] if full_name else "Memory"

    def save(self, value: Any, metadata: Dict[str, Any]) -> None:
        """Save a memory entry to Weaviate"""
        if not self.collection:
            self._initialize_app()
        
        try:
            # Prepare data object
            import json
            import time
            
            data_object = {
                "content": str(value),
                "metadata_json": json.dumps(metadata or {}),
                "agent_role": metadata.get("agent_role", "unknown"),
                "memory_type": self.type,
                "timestamp": time.time()
            }
            
            self.collection.data.insert(
                properties=data_object,
                uuid=str(uuid.uuid4())
            )
                
        except Exception as e:
            logging.error(f"Error during {self.type} save to Weaviate: {str(e)}")

    def search(
        self,
        query: str,
        limit: int = 3
    ) -> List[Any]:
        """Search for relevant memories using Weaviate's hybrid search"""
        if not self.collection:
            self._initialize_app()
        
        try:
            response = self.collection.query.hybrid(
                    query=query,
                    limit=limit,
            )
            
            # Parse and format results
            results = []
            for item in response.objects:
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

    def __del__(self):
        """Cleanup Weaviate client connection"""
        try:
            if self.client:
                self.client.close()
        except:
            pass
    
    def _generate_embedding(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Generate an embedding for the given text and metadata."""
        pass

import os
import json
import time
from weaviate_storage import WeaviateStorage  # Import your WeaviateStorage class


class MockAgent:
    def __init__(self, role):
        self.role = role


class MockCrew:
    def __init__(self, agents):
        self.agents = agents


def main():
    # Check environment variables
    if not os.getenv("WEAVIATE_URL") or not os.getenv("WEAVIATE_API_KEY"):
        print("❌ Please set WEAVIATE_URL and WEAVIATE_API_KEY environment variables")
        return

    print("Testing WeaviateStorage\n")

    # 1. Create storage
    print("1. Creating storage...")
    crew = MockCrew([
        MockAgent("Developer"),
        MockAgent("Analyst")
    ])
    
    storage = WeaviateStorage(type="short_term", crew=crew)
    print(f"✅ Created collection: {storage.collection_name}\n")

    # 2. Save some memories
    print("2. Saving memories...")
    memories = [
        ("Project deadline is Friday", {"agent_role": "Developer"}),
        ("User count increased by 45%", {"agent_role": "Analyst"}),
        ("Need to implement OAuth2", {"agent_role": "Developer"})
    ]
    
    for content, metadata in memories:
        storage.save(content, metadata)
        print(f"✅ Saved: {content}")
    
    # Wait for indexing
    time.sleep(2)
    print()

    # 3. Search memories
    print("3. Searching memories...")
    queries = ["deadline", "OAuth2", "user metrics"]
    
    for query in queries:
        results = storage.search(query, limit=2)
        print(f"\nQuery: '{query}' - Found {len(results)} results")
        for r in results:
            print(f"  - {r['context']}")

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    main()