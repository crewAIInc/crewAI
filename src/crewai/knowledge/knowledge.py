from typing import List, Optional

from pydantic import BaseModel

from .embedder.base_embedder import BaseEmbedder
from .embedder.fastembed import FastEmbed
from .source.base_knowledge_source import BaseKnowledgeSource


class Knowledge(BaseModel):
    sources: Optional[List[BaseKnowledgeSource]] = None
    embedder: BaseEmbedder

    def __init__(
        self,
        sources: Optional[List[BaseKnowledgeSource]] = None,
        embedder: Optional[BaseEmbedder] = None,
    ):
        super().__init__()
        self.sources = sources or []
        self.embedder = embedder or FastEmbed()
