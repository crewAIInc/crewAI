from typing import List, Optional

from pydantic import Field

from crewai.knowledge.source.base_knowledge_source import BaseKnowledgeSource
from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage
from crewai.utilities.logger import Logger


class StringKnowledgeSource(BaseKnowledgeSource):
    """A knowledge source that stores and queries plain text content using embeddings."""

    _logger: Logger = Logger(verbose=True)
    content: str = Field(...)
    collection_name: Optional[str] = Field(default=None)

    def model_post_init(self, _) -> None:
        """Post-initialization method to validate content and initialize storage.
        
        This method is called after the model is initialized to perform content validation
        and set up the knowledge storage system. It ensures that:
        1. The content is a valid string
        2. The storage system is properly initialized
        
        Raises:
            ValueError: If content validation fails or storage initialization fails
        """
        try:
            self.validate_content()
            if self.storage is None:
                self.storage = KnowledgeStorage(collection_name=self.collection_name)
            self.storage.initialize_knowledge_storage()
        except Exception as e:
            error_msg = f"Failed to initialize knowledge storage: {str(e)}"
            self._logger.log("error", error_msg, "red")
            raise ValueError(error_msg)

    def validate_content(self) -> None:
        """Validate that the content is a valid string.
        
        Raises:
            ValueError: If content is not a string or is empty
        """
        if not isinstance(self.content, str) or not self.content.strip():
            error_msg = "StringKnowledgeSource only accepts string content"
            self._logger.log("error", error_msg, "red")
            raise ValueError(error_msg)

    def add(self) -> None:
        """Add string content to the knowledge source, chunk it, compute embeddings, and save them.
        
        This method processes the content by:
        1. Chunking the text into smaller pieces
        2. Adding the chunks to the source
        3. Computing embeddings and saving them
        
        Raises:
            ValueError: If storage is not initialized when trying to save documents
        """
        new_chunks = self._chunk_text(self.content)
        self.chunks.extend(new_chunks)
        self._save_documents()

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks based on chunk_size and chunk_overlap.
        
        Args:
            text: The text to split into chunks
            
        Returns:
            List[str]: List of text chunks
        """
        return [
            text[i : i + self.chunk_size]
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
        ]
