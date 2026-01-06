"""NVIDIA embedding callable implementation."""

from typing import cast

import httpx
import numpy as np

from crewai.rag.core.base_embeddings_callable import EmbeddingFunction
from crewai.rag.core.types import Documents, Embeddings


class NvidiaEmbeddingFunction(EmbeddingFunction[Documents]):
    """NVIDIA embedding function using the /v1/embeddings endpoint.

    Supports NVIDIA's embedding models through the OpenAI-compatible API.
    Default base URL: https://integrate.api.nvidia.com/v1
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "nvidia/nv-embed-v1",
        api_base: str = "https://integrate.api.nvidia.com/v1",
        input_type: str = "query",
        truncate: str = "NONE",
        **kwargs: dict,
    ) -> None:
        """Initialize NVIDIA embedding function.

        Args:
            api_key: NVIDIA API key
            model_name: NVIDIA embedding model name (e.g., 'nvidia/nv-embed-v1')
            api_base: Base URL for NVIDIA API
            input_type: Type of input for asymmetric models ('query' or 'passage')
                       - 'query': For search queries or questions
                       - 'passage': For documents/passages to be searched
            truncate: Truncation strategy ('NONE', 'START', 'END')
            **kwargs: Additional parameters
        """
        self._api_key = api_key
        self._model_name = model_name
        self._api_base = api_base.rstrip("/")
        self._input_type = input_type
        self._truncate = truncate
        self._session = httpx.Client()

        # Models that require input_type parameter
        self._requires_input_type = any(
            keyword in model_name.lower()
            for keyword in ["embedqa", "embedcode", "nemoretriever"]
        )

    @staticmethod
    def name() -> str:
        """Return the name of the embedding function for ChromaDB compatibility."""
        return "nvidia"

    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings for the given documents.

        Args:
            input: List of documents to embed

        Returns:
            List of embedding vectors as numpy arrays
        """
        # Build request payload
        payload = {
            "model": self._model_name,
            "input": input,
        }

        # Add input_type and truncate for models that require them
        if self._requires_input_type:
            payload["input_type"] = self._input_type
            payload["truncate"] = self._truncate

        # NVIDIA embeddings API (OpenAI-compatible)
        response = self._session.post(
            f"{self._api_base}/embeddings",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60.0,
        )

        # Handle errors
        if response.status_code != 200:
            error_detail = ""
            try:
                error_data = response.json()
                error_detail = error_data.get("detail", "") or error_data.get("error", {}).get("message", "")
            except Exception:
                error_detail = response.text[:500]

            raise RuntimeError(
                f"NVIDIA embeddings API returned status {response.status_code}: {error_detail}"
            )

        # Parse response
        result = response.json()
        embeddings_data = result.get("data", [])

        if not embeddings_data:
            raise ValueError(f"No embeddings returned from NVIDIA API for {len(input)} documents")

        # Sort by index and extract embeddings
        embeddings_data = sorted(embeddings_data, key=lambda x: x.get("index", 0))
        embeddings = [np.array(item["embedding"], dtype=np.float32) for item in embeddings_data]

        return cast(Embeddings, embeddings)

    def __del__(self) -> None:
        """Clean up HTTP session."""
        if hasattr(self, "_session"):
            self._session.close()
