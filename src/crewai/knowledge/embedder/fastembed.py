from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from .base_embedder import BaseEmbedder

try:
    from fastembed_gpu import TextEmbedding  # type: ignore

    FASTEMBED_AVAILABLE = True
except ImportError:
    try:
        from fastembed import TextEmbedding

        FASTEMBED_AVAILABLE = True
    except ImportError:
        FASTEMBED_AVAILABLE = False


class FastEmbed(BaseEmbedder):
    """
    A wrapper class for text embedding models using FastEmbed
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the embedding model

        Args:
            model_name: Name of the model to use
            cache_dir: Directory to cache the model
            gpu: Whether to use GPU acceleration
        """
        if not FASTEMBED_AVAILABLE:
            raise ImportError(
                "FastEmbed is not installed. Please install it with: "
                "uv pip install fastembed or uv pip install fastembed-gpu for GPU support"
            )

        self.model = TextEmbedding(
            model_name=model_name,
            cache_dir=str(cache_dir) if cache_dir else None,
        )

    def embed_chunks(self, chunks: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for a list of text chunks

        Args:
            chunks: List of text chunks to embed

        Returns:
            List of embeddings
        """
        embeddings = list(self.model.embed(chunks))
        return embeddings

    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for a list of texts

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings
        """
        embeddings = list(self.model.embed(texts))
        return embeddings

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text

        Args:
            text: Text to embed

        Returns:
            Embedding array
        """
        return self.embed_texts([text])[0]

    @property
    def dimension(self) -> int:
        """Get the dimension of the embeddings"""
        # Generate a test embedding to get dimensions
        test_embed = self.embed_text("test")
        return len(test_embed)
