import os

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, ConfigDict, Field

from crewai.knowledge.source.base_knowledge_source import BaseKnowledgeSource
from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage
from crewai.utilities.logger import Logger
from crewai.utilities.constants import DEFAULT_SCORE_THRESHOLD

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # removes logging from fastembed


class Knowledge(BaseModel):
    """
    Knowledge is a collection of sources and setup for the vector store to save and query relevant context.
    Args:
        sources: List[BaseKnowledgeSource] = Field(default_factory=list)
        storage: KnowledgeStorage = Field(default_factory=KnowledgeStorage)
        embedder_config: Optional[Dict[str, Any]] = None
    """

    sources: List[BaseKnowledgeSource] = Field(default_factory=list)
    model_config = ConfigDict(arbitrary_types_allowed=True)
    storage: KnowledgeStorage = Field(default_factory=KnowledgeStorage)
    embedder_config: Optional[Dict[str, Any]] = None
    store_dir: Optional[str] = None

    def __init__(
        self,
        store_dir: str,
        embedder_config: Optional[Dict[str, Any]] = None,
        storage: Optional[KnowledgeStorage] = None,
        **data,
    ):
        super().__init__(**data)
        if storage:
            self.storage = storage
        else:
            self.storage = KnowledgeStorage(
                embedder_config=embedder_config, store_dir=store_dir
            )

    def query(
        self, query: List[str], limit: int = 3, preference: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Query across all knowledge sources to find the most relevant information.
        Returns the top_k most relevant chunks.
        """

        results = self.storage.search(
            query,
            limit,
            filter={"preference": preference} if preference else None,
            score_threshold=DEFAULT_SCORE_THRESHOLD,
        )
        return results
