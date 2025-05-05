"""Example of using a custom storage with CrewAI."""

from pathlib import Path

import chromadb
from chromadb.config import Settings

from crewai import Agent, Crew, Task
from crewai.knowledge.source.custom_storage_knowledge_source import (
    CustomStorageKnowledgeSource,
)
from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage


class CustomKnowledgeStorage(KnowledgeStorage):
    """Custom knowledge storage that uses a specific persistent directory.
    
    Args:
        persist_directory (str): Path to the directory where ChromaDB will persist data.
        embedder: Embedding function to use for the collection. Defaults to None.
        collection_name (str, optional): Name of the collection. Defaults to None.
    
    Raises:
        ValueError: If persist_directory is empty or invalid.
    """
    
    def __init__(self, persist_directory: str, embedder=None, collection_name=None):
        if not persist_directory:
            raise ValueError("persist_directory cannot be empty")
        self.persist_directory = persist_directory
        super().__init__(embedder=embedder, collection_name=collection_name)
    
    def initialize_knowledge_storage(self):
        """Initialize the knowledge storage with a custom persistent directory.
        
        Creates a ChromaDB PersistentClient with the specified directory and
        initializes a collection with the provided name and embedding function.
        
        Raises:
            Exception: If collection creation or retrieval fails.
        """
        try:
            chroma_client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(allow_reset=True),
            )
            self.app = chroma_client
            
            collection_name = (
                "knowledge" if not self.collection_name else self.collection_name
            )
            self.collection = self.app.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedder_config,
            )
        except Exception as e:
            raise Exception(f"Failed to create or get collection: {e}")


def get_knowledge_source_with_custom_storage(
    folder_name: str, 
    embedder=None
) -> CustomStorageKnowledgeSource:
    """Create a knowledge source with a custom storage.
    
    Args:
        folder_name (str): Name of the folder to store embeddings and collection.
        embedder: Embedding function to use. Defaults to None.
        
    Returns:
        CustomStorageKnowledgeSource: Configured knowledge source with custom storage.
        
    Raises:
        Exception: If storage initialization fails.
    """
    try:
        persist_path = f"vectorstores/knowledge_{folder_name}"
        storage = CustomKnowledgeStorage(
            persist_directory=persist_path,
            embedder=embedder,
            collection_name=folder_name
        )
        
        storage.initialize_knowledge_storage()
        
        source = CustomStorageKnowledgeSource(collection_name=folder_name)
        source.storage = storage
        
        source.validate_content()
        
        return source
    except Exception as e:
        raise Exception(f"Failed to initialize knowledge source: {e}")


def main() -> None:
    """Example of using a custom storage with CrewAI.
    
    This function demonstrates how to:
    1. Create a knowledge source with pre-existing embeddings
    2. Use it with a Crew
    3. Run the Crew to perform tasks
    """
    try:
        knowledge_source = get_knowledge_source_with_custom_storage(folder_name="example")
        
        agent = Agent(role="test", goal="test", backstory="test")
        task = Task(description="test", expected_output="test", agent=agent)
        
        crew = Crew(
            agents=[agent],
            tasks=[task],
            knowledge_sources=[knowledge_source]
        )
        
        result = crew.kickoff()
        print(result)
    except Exception as e:
        print(f"Error running example: {e}")


if __name__ == "__main__":
    main()
