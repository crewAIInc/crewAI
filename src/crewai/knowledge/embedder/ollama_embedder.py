import os
from typing import List, Optional

import numpy as np
from openai import OpenAI

from .base_embedder import BaseEmbedder


class OllamaEmbedder(BaseEmbedder):
    """
    A wrapper class for text embedding models using Ollama's API
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: str = "http://localhost:11434/v1",
    ):
        """
        Initialize the embedding model

        Args:
            model_name: Name of the model to use
            api_key: API key (defaults to 'ollama' or environment variable 'OLLAMA_API_KEY')
            base_url: Base URL for the Ollama API (default is 'http://localhost:11434/v1')
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OLLAMA_API_KEY") or "ollama"
        self.base_url = base_url
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

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
        max_batch_size = 2048  # Adjust batch size if necessary
        for i in range(0, len(texts), max_batch_size):
            batch = texts[i : i + max_batch_size]
            response = self.client.embeddings.create(input=batch, model=self.model_name)
            batch_embeddings = [np.array(item.embedding) for item in response.data]
            embeddings.extend(batch_embeddings)
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
        # Embedding dimensions may vary; we'll determine it dynamically
        test_embed = self.embed_text("test")
        return len(test_embed)
