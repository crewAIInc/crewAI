from abc import ABC, abstractmethod
from typing import List

import numpy as np


class BaseEmbedder(ABC):
    """
    Abstract base class for text embedding models
    """

    @abstractmethod
    def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of text chunks

        Args:
            chunks: List of text chunks to embed

        Returns:
            Array of embeddings
        """
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts

        Args:
            texts: List of texts to embed

        Returns:
            Array of embeddings
        """
        pass

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text

        Args:
            text: Text to embed

        Returns:
            Embedding array
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get the dimension of the embeddings"""
        pass
