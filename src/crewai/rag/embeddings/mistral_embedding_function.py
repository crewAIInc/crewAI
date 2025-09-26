"""
Mistral AI embedding function for CrewAI.

This module provides integration with Mistral AI's embedding API
for use with CrewAI's RAG (Retrieval-Augmented Generation) functionality.
"""

import os
from typing import cast

import requests
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.api.types import validate_embedding_function


class MistralEmbeddingFunction(EmbeddingFunction):
    """
    Mistral AI embedding function compatible with ChromaDB.

    This class implements the ChromaDB EmbeddingFunction interface
    to provide seamless integration with Mistral AI's embedding models.

    Attributes:
        api_key (str): Mistral API key for authentication
        model_name (str): Name of the Mistral embedding model to use
        base_url (str): Base URL for Mistral API endpoints
        max_retries (int): Maximum number of retries for failed requests
        timeout (int): Request timeout in seconds
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "mistral-embed",
        base_url: str = "https://api.mistral.ai/v1",
        max_retries: int = 3,
        timeout: int = 30,
    ):
        """
        Initialize Mistral embedding function.

        Args:
            api_key: Mistral API key (defaults to MISTRAL_API_KEY env var)
            model_name: Mistral embedding model name
            base_url: Mistral API base URL
            max_retries: Maximum number of retries for API calls
            timeout: Request timeout in seconds

        Raises:
            ValueError: If API key is not provided or invalid
        """
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Mistral API key is required. Set MISTRAL_API_KEY environment variable."
            )

        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.timeout = timeout

        # Validate the embedding function
        try:
            validate_embedding_function(self)
        except Exception as e:
            raise ValueError(f"Invalid Mistral embedding function: {e!s}") from e

    def __call__(self, input: Documents) -> Embeddings:
        """
        Generate embeddings for input documents.

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
            raise ValueError("Expected Embeddings to be non-empty list or numpy array")

        # Prepare the request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {"model": self.model_name, "input": input}

        # Make API request with retry logic
        # Ensure at least one attempt is made, even if max_retries is 0
        attempts = max(1, self.max_retries)

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

            except requests.exceptions.RequestException as e:
                # If this is the last attempt, raise the error
                if attempt == attempts - 1:
                    raise RuntimeError(
                        f"Failed to get embeddings from Mistral API after "
                        f"{attempts} attempts: {e!s}"
                    ) from e
                # Otherwise, continue to next attempt
                continue

        # This should never be reached, but added for type safety
        raise RuntimeError("Unexpected end of retry loop")

    def get_model_info(self) -> dict:
        """
        Get information about the current model.

        Returns:
            Dictionary containing model information
        """
        return {
            "provider": "mistral",
            "model": self.model_name,
            "base_url": self.base_url,
        }
