from typing import List, Dict, Any, Optional
import faiss
import numpy as np
from pydantic import BaseModel, Field

from crewai.tools import BaseTool
from crewai.utilities import EmbeddingConfigurator

class FAISSSearchTool(BaseTool):
    name: str = "FAISS Search Tool"
    description: str = "Search through documents using FAISS vector similarity search"
    
    def __init__(
        self,
        index_type: str = "L2",
        dimension: int = 384,  # Default for BAAI/bge-small-en-v1.5
        embedder_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.embedder_config = embedder_config
        self.dimension = dimension
        self.index = self._create_index(index_type)
        self.texts = []
        self._initialize_embedder()

    def _create_index(self, index_type: str) -> faiss.Index:
        if index_type == "L2":
            return faiss.IndexFlatL2(self.dimension)
        elif index_type == "IP":
            return faiss.IndexFlatIP(self.dimension)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")

    def _initialize_embedder(self):
        configurator = EmbeddingConfigurator()
        self.embedder = configurator.configure_embedder(self.embedder_config)

    def _run(
        self,
        query: str,
        k: int = 3,
        score_threshold: float = 0.6
    ) -> List[Dict[str, Any]]:
        query_embedding = self.embedder.embed_text(query)
        D, I = self.index.search(
            np.array([query_embedding], dtype=np.float32),
            k
        )
        
        results = []
        for i, (dist, idx) in enumerate(zip(D[0], I[0])):
            if idx < len(self.texts):
                score = 1.0 / (1.0 + dist)  # Convert distance to similarity score
                if score >= score_threshold:
                    results.append({
                        "text": self.texts[idx],
                        "score": score
                    })
        return results

    def add_texts(self, texts: List[str]) -> None:
        embeddings = self.embedder.embed_texts(texts)
        self.index.add(np.array(embeddings, dtype=np.float32))
        self.texts.extend(texts)
