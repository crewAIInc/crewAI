"""Example of using a custom storage with CrewAI."""

import chromadb
from chromadb.config import Settings
from crewai import Agent, Crew, Task
from crewai.knowledge.source.custom_storage_knowledge_source import CustomStorageKnowledgeSource
from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage


class CustomKnowledgeStorage(KnowledgeStorage):
    """Custom knowledge storage that uses a specific persistent directory."""
    
    def __init__(self, persist_directory: str, embedder=None, collection_name=None):
        self.persist_directory = persist_directory
        super().__init__(embedder=embedder, collection_name=collection_name)
    
    def initialize_knowledge_storage(self):
        """Initialize the knowledge storage with a custom persistent directory."""
        chroma_client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(allow_reset=True),
        )
        self.app = chroma_client
        try:
            collection_name = (
                "knowledge" if not self.collection_name else self.collection_name
            )
            self.collection = self.app.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedder_config,
            )
        except Exception as e:
            raise Exception(f"Failed to create or get collection: {e}")


def get_knowledge_source_with_custom_storage(folder_name: str, embedder=None):
    """Create a knowledge source with a custom storage."""
    persist_path = f"vectorstores/knowledge_{folder_name}"
    storage = CustomKnowledgeStorage(
        persist_directory=persist_path,
        embedder=embedder,
        collection_name=folder_name
    )
    
    storage.initialize_knowledge_storage()
    
    source = CustomStorageKnowledgeSource(collection_name=folder_name)
    
    source.storage = storage
    
    return source


def main():
    """Example of using a custom storage with CrewAI."""
    knowledge_source = get_knowledge_source_with_custom_storage(folder_name="example")
    
    agent = Agent(role="test", goal="test", backstory="test")
    task = Task(description="test", agent=agent)
    
    crew = Crew(
        agents=[agent],
        tasks=[task],
        knowledge_sources=[knowledge_source]
    )
    
    result = crew.kickoff()
    print(result)


if __name__ == "__main__":
    main()
