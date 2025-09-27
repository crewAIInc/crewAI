"""
Mistral AI embedding function for CrewAI.

This module provides integration with Mistral AI's embedding API
for use with CrewAI's RAG (Retrieval-Augmented Generation) functionality.
"""

import json
from typing import cast

import requests
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.api.types import validate_embedding_function


class MistralEmbeddingFunction(EmbeddingFunction):
    """Mistral AI embedding function implementation."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "mistral-embed",
        base_url: str = "https://api.mistral.ai/v1",
        timeout: int = 30,
        max_retries: int = 3,
    ):
        """Initialize Mistral embedding function.
        
        Args:
            api_key: Mistral API key
            model_name: Model name to use for embeddings
            base_url: Base URL for Mistral API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Validate the embedding function
        validate_embedding_function(self)

    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings for input documents.
        
        Args:
            input: Documents to embed (string or list of strings)
            
        Returns:
            List of embedding vectors
            
        Raises:
            RuntimeError: If API calls fail after max_retries attempts
        """
        if isinstance(input, str):
            input = [input]

        if not input:
            raise ValueError("Expected Documents to be non-empty list or string")

        # Prepare the request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {"model": self.model_name, "input": input}

        # Make API request with retry logic
        attempts = 1 + self.max_retries
        last_exception = None

        for attempt in range(attempts):
            try:
                response = requests.post(
                    f"{self.base_url}/embeddings",
                    headers=headers,
                    json=data,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                
                result = response.json()
                embeddings = [item["embedding"] for item in result["data"]]
                return cast(Embeddings, embeddings)
                
            except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
                last_exception = e
                if attempt == attempts - 1:
                    break

        # If we get here, all attempts failed
        raise RuntimeError(
            f"Failed to get embeddings from Mistral API after "
            f"{attempts} attempts: {last_exception!s}"
        ) from last_exception