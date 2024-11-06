from typing import List

from pydantic import BaseModel, ConfigDict, Field

from crewai.knowledge.embedder.base_embedder import BaseEmbedder
from crewai.knowledge.embedder.fastembed import FastEmbed
from crewai.knowledge.source.base_knowledge_source import BaseKnowledgeSource


class Knowledge(BaseModel):
    sources: List[BaseKnowledgeSource] = Field(default_factory=list)
    embedder: BaseEmbedder = Field(default_factory=FastEmbed)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        super().__init__(**data)
        # Call add on all sources during initialization
        for source in self.sources:
            source.add(self.embedder)

    def query(self, query: str, top_k: int = 3) -> List[str]:
        """
        Query across all knowledge sources to find the most relevant information.
        Returns the top_k most relevant chunks.
        """
        if not self.sources:
            return []

        # Collect all chunks and embeddings from all sources
        all_chunks = []
        all_embeddings = []

        for source in self.sources:
            all_chunks.extend(source.chunks)
            all_embeddings.extend(source.get_embeddings())

        # Embed the query
        query_embedding = self.embedder.embed_text(query)

        # Calculate similarities
        similarities = []
        for idx, embedding in enumerate(all_embeddings):
            similarity = query_embedding.dot(embedding)
            similarities.append((similarity, idx))

        # Sort by similarity
        similarities.sort(reverse=True, key=lambda x: x[0])

        # Get top_k results
        top_chunks = [all_chunks[idx] for _, idx in similarities[:top_k]]

        return top_chunks
