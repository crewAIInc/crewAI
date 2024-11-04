from typing import List, Optional, Union
from pathlib import Path
import numpy as np

try:
    from fastembed_gpu import TextEmbedding

    FASTEMBED_AVAILABLE = True
except ImportError:
    try:
        from fastembed import TextEmbedding

        FASTEMBED_AVAILABLE = True
    except ImportError:
        FASTEMBED_AVAILABLE = False


class Embeddings:
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
                "pip install fastembed or pip install fastembed-gpu for GPU support"
            )

        self.model = TextEmbedding(
            model_name=model_name,
            cache_dir=str(cache_dir) if cache_dir else None,
        )

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts

        Args:
            texts: List of texts to embed

        Returns:
            Array of embeddings
        """
        # FastEmbed returns a generator, convert to list then numpy array
        embeddings = list(self.model.embed(texts))
        return np.array(embeddings)

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
