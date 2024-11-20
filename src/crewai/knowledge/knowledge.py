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

    def __init__(self, embedder_config: Optional[Dict[str, Any]] = None, **data):
        super().__init__(**data)
        self.storage = KnowledgeStorage(embedder_config=embedder_config or None)

        try:
            for source in self.sources:
                source.add()
        except Exception as e:
            Logger(verbose=True).log(
                "warning",
                f"Failed to init knowledge: {e}",
                color="yellow",
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
