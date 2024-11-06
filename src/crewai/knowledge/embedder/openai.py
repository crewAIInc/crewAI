import os
from typing import List, Optional

import numpy as np
from openai import OpenAI

from .base_embedder import BaseEmbedder


class OpenAIEmbedder(BaseEmbedder):
    """
    A wrapper class for text embedding models using OpenAI's Embedding API
    """

    def __init__(
        self,
        model_name: str = "text-embedding-ada-002",
        api_key: Optional[str] = None,
    ):
        """
        Initialize the embedding model

        Args:
            model_name: Name of the model to use
            api_key: OpenAI API key
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key must be provided or set in the environment variable 'OPENAI_API_KEY'"
            )
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="http://localhost:11434/v1",
        )

    def embed_chunks(self, chunks: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for a list of text chunks

        Args:
            chunks: List of text chunks to embed

        Returns:
            List of embeddings
        """
        return self.embed_texts(chunks)

    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for a list of texts

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings
        """
        embeddings = []
        max_batch_size = 2048  # OpenAI recommends smaller batch sizes
        for i in range(0, len(texts), max_batch_size):
            batch = texts[i : i + max_batch_size]
            response = self.client.embeddings.create(input=batch, model=self.model_name)
            batch_embeddings = [np.array(data.embedding) for data in response.data]
            embeddings.extend(batch_embeddings)
        return embeddings

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding fors a single text

        Args:
            text: Text to embed

        Returns:
            Embedding array
        """
        return self.embed_texts([text])[0]

    @property
    def dimension(self) -> int:
        """Get the dimension of the embeddings"""
        # For OpenAI's text-embedding-ada-002, the dimension is 1536
        return 1536
