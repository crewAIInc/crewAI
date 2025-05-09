import logging
from typing import Optional

from pydantic import Field

from crewai.knowledge.source.base_knowledge_source import BaseKnowledgeSource
from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage

logger = logging.getLogger(__name__)


class CustomStorageKnowledgeSource(BaseKnowledgeSource):
    """A knowledge source that uses a pre-existing storage with embeddings.
    
    This class allows users to use pre-existing vector embeddings without re-embedding
    when using CrewAI. It acts as a bridge between BaseKnowledgeSource and KnowledgeStorage.
    
    Args:
        collection_name (Optional[str]): Name of the collection in the vector database.
            Defaults to None.
    
    Attributes:
        storage (KnowledgeStorage): The underlying storage implementation that contains
            the pre-existing embeddings.
    """

    collection_name: Optional[str] = Field(default=None)

    def validate_content(self):
        """Validates that the storage is properly initialized.
        
        Raises:
            ValueError: If storage is not initialized before use.
        """
        if not hasattr(self, 'storage') or self.storage is None:
            raise ValueError("Storage not initialized. Please set storage before use.")
        logger.debug(f"Storage validated for collection: {self.collection_name}")

    def add(self) -> None:
        """No need to add content as we're using pre-existing storage.
        
        This method is intentionally empty as the embeddings already exist in the storage.
        """
        logger.debug(f"Skipping add operation for pre-existing storage: {self.collection_name}")
        pass
