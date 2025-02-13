import logging
import re
from typing import List, Dict, Any, Optional

import faiss
import numpy as np
from pydantic import BaseModel, Field

from crewai.tools import BaseTool
from crewai.utilities import EmbeddingConfigurator

logger = logging.getLogger(__name__)

class FAISSSearchTool(BaseTool):
    """FAISS vector similarity search tool for efficient document search."""
    
    model_config = {"extra": "allow"}
    
    name: str = "FAISS Search Tool"
    description: str = "Search through documents using FAISS vector similarity search"
    embedder_config: Optional[Dict[str, Any]] = Field(default=None)
    dimension: int = Field(default=384)  # Default for BAAI/bge-small-en-v1.5
    texts: List[str] = Field(default_factory=list)
    index_type: str = Field(default="L2")
    index: Any = Field(default=None)  # FAISS index instance
    embedder: Any = Field(default=None)  # Embedder instance
    
    def __init__(
        self,
        index_type: str = "L2",
        dimension: int = 384,
        embedder_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize FAISS search tool.
        
        Args:
            index_type: Type of FAISS index ("L2" or "IP")
            dimension: Embedding dimension
            embedder_config: Configuration for the embedder
        """
        super().__init__()
        self.dimension = dimension
        self.embedder_config = embedder_config
        self.index_type = index_type
        self.index = self._create_index(index_type)
        self._initialize_embedder()

    def _create_index(self, index_type: str) -> faiss.Index:
        """Create FAISS index of specified type.
        
        Args:
            index_type: Type of index ("L2" or "IP")
            
        Returns:
            FAISS index instance
            
        Raises:
            ValueError: If index_type is not supported
        """
        if index_type == "L2":
            return faiss.IndexFlatL2(self.dimension)
        elif index_type == "IP":
            return faiss.IndexFlatIP(self.dimension)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")

    def _initialize_embedder(self):
        """Initialize the embedder using the provided configuration."""
        from crewai.knowledge.embedder.fastembed import FastEmbed
        self.embedder = FastEmbed()

    def _sanitize_query(self, query: str) -> str:
        """Remove potentially harmful characters from query.
        
        Args:
            query: Input query string
            
        Returns:
            Sanitized query string
        """
        return re.sub(r'[^\w\s]', '', query)

    def _run(
        self,
        query: str,
        k: int = 3,
        score_threshold: float = 0.6
    ) -> List[Dict[str, Any]]:
        """Search for similar texts using FAISS.
        
        Args:
            query: Search query
            k: Number of results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of dictionaries containing matched texts and scores
            
        Raises:
            ValueError: If input parameters are invalid
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")
        if k < 1:
            raise ValueError("k must be positive")
        if not 0 <= score_threshold <= 1:
            raise ValueError("score_threshold must be between 0 and 1")
            
        logger.debug(f"Searching for query: {query} with k={k}")
        query = self._sanitize_query(query)
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
        """Add texts to the search index.
        
        Args:
            texts: List of texts to add
            
        Raises:
            ValueError: If embedding or indexing fails
        """
        try:
            embeddings = self.embedder.embed_texts(texts)
            self.index.add(np.array(embeddings, dtype=np.float32))
            self.texts.extend(texts)
        except Exception as e:
            raise ValueError(f"Failed to add texts: {str(e)}")
            
    def add_texts_batch(self, texts: List[str], batch_size: int = 1000) -> None:
        """Add texts in batches to prevent memory issues.
        
        Args:
            texts: List of texts to add
            batch_size: Size of each batch
            
        Raises:
            ValueError: If batch_size is invalid
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
            
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            self.add_texts(batch)
            
    def clear_index(self) -> None:
        """Clear the index and stored texts."""
        self.index = self._create_index(self.index_type)
        self.texts = []
        
    @property
    def index_size(self) -> int:
        """Return number of vectors in index."""
        return len(self.texts)
        
    @property
    def is_empty(self) -> bool:
        """Check if index is empty."""
        return len(self.texts) == 0
