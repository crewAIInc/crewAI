from typing import List, Optional, Dict, Any

from pydantic import BaseModel, ConfigDict, Field

from crewai.knowledge.embedder.base_embedder import BaseEmbedder
from crewai.knowledge.embedder.fastembed import FastEmbed
from crewai.knowledge.source.base_knowledge_source import BaseKnowledgeSource
from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage


class Knowledge(BaseModel):
    sources: List[BaseKnowledgeSource] = Field(default_factory=list)
    embedder: BaseEmbedder = Field(default_factory=FastEmbed)

    model_config = ConfigDict(arbitrary_types_allowed=True)
    agents: List[str] = Field(default_factory=list)
    storage: KnowledgeStorage = Field(default_factory=KnowledgeStorage)

    def __init__(self, **data):
        super().__init__(**data)
        # Call add on all sources during initialization
        for source in self.sources:
            source.add(self.embedder)

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
            score_threshold=0.35,
        )
        return results
