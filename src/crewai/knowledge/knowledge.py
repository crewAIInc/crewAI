import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from crewai.knowledge.source.base_knowledge_source import BaseKnowledgeSource
from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # removes logging from fastembed


class Knowledge(BaseModel):
    """
    Knowledge is a collection of sources and setup for the vector store to save and query relevant context.
    Args:
        sources: List[BaseKnowledgeSource] = Field(default_factory=list)
        storage: Optional[KnowledgeStorage] = Field(default=None)
        embedder_config: Optional[Dict[str, Any]] = None
    """

    sources: List[BaseKnowledgeSource] = Field(default_factory=list)
    model_config = ConfigDict(arbitrary_types_allowed=True)
    storage: Optional[KnowledgeStorage] = Field(default=None)
    embedder_config: Optional[Dict[str, Any]] = None
    collection_name: Optional[str] = None

    def __init__(
        self,
        collection_name: str,
        sources: List[BaseKnowledgeSource],
        embedder_config: Optional[Dict[str, Any]] = None,
        storage: Optional[KnowledgeStorage] = None,
        **data,
    ):
        super().__init__(**data)
        if storage:
            self.storage = storage
        else:
            self.storage = KnowledgeStorage(
                embedder_config=embedder_config, collection_name=collection_name
            )
        self.sources = sources
        self.storage.initialize_knowledge_storage()
        for source in sources:
            source.storage = self.storage
            source.add()

    def query(self, query: List[str], limit: int = 3) -> List[Dict[str, Any]]:
        """
        Query across all knowledge sources to find the most relevant information.
        Returns the top_k most relevant chunks.
        
        Raises:
            ValueError: If storage is not initialized.
        """
        if self.storage is None:
            raise ValueError("Storage is not initialized.")
            
        results = self.storage.search(
            query,
            limit,
        )
        return results

    def _add_sources(self):
        for source in self.sources:
            source.storage = self.storage
            source.add()
