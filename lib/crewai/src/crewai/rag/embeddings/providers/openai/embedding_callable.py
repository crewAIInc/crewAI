"""OpenAI embedding function implementation."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from crewai.rag.core.base_embeddings_callable import EmbeddingFunction
from crewai.rag.core.types import Documents, Embeddings


class OpenAIEmbeddingFunction(EmbeddingFunction[Documents]):
    """Embedding function backed by the OpenAI SDK.

    Also serves Azure OpenAI (``api_type="azure"``) and OpenAI-compatible
    endpoints such as OpenRouter (via ``api_base``).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "text-embedding-ada-002",
        organization_id: str | None = None,
        api_base: str | None = None,
        api_type: str | None = None,
        api_version: str | None = None,
        deployment_id: str | None = None,
        default_headers: dict[str, Any] | None = None,
        dimensions: int | None = None,
    ) -> None:
        """Initialize the OpenAI embedding function.

        Args:
            api_key: API key. Falls back to the OPENAI_API_KEY environment variable.
            model_name: Embedding model name.
            organization_id: OpenAI organization ID.
            api_base: Base URL of the API (the endpoint for Azure).
            api_type: Set to "azure" to use Azure OpenAI.
            api_version: Azure API version. Required for Azure.
            deployment_id: Azure deployment name. Required for Azure.
            default_headers: Headers sent with every request.
            dimensions: Output dimensions. Only sent for text-embedding-3 models.
        """
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "openai is required for openai embeddings. "
                "Install it with: uv add openai"
            ) from e

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("The OPENAI_API_KEY environment variable is not set.")

        self.model_name = model_name
        self.organization_id = organization_id
        self.api_base = api_base
        self.api_type = api_type
        self.api_version = api_version
        self.deployment_id = deployment_id
        self.default_headers = default_headers
        self.dimensions = dimensions

        if api_type == "azure":
            if api_version is None:
                raise ValueError("api_version must be specified for Azure OpenAI")
            if deployment_id is None:
                raise ValueError("deployment_id must be specified for Azure OpenAI")
            if api_base is None:
                raise ValueError("api_base must be specified for Azure OpenAI")

            self.client: Any = openai.AzureOpenAI(
                api_key=self.api_key,
                api_version=api_version,
                azure_endpoint=api_base,
                azure_deployment=deployment_id,
                default_headers=default_headers,
            )
            return

        client_params: dict[str, Any] = {"api_key": self.api_key}
        if organization_id is not None:
            client_params["organization"] = organization_id
        if api_base is not None:
            client_params["base_url"] = api_base
        if default_headers is not None:
            client_params["default_headers"] = default_headers
        self.client = openai.OpenAI(**client_params)

    @staticmethod
    def name() -> str:
        """Return the name of the embedding function for ChromaDB compatibility."""
        return "openai"

    def __call__(self, input: Documents) -> Embeddings:
        """Embed documents with a single embeddings request.

        Args:
            input: Documents to embed.

        Returns:
            One float32 vector per document.

        Raises:
            ValueError: If no documents are given.
        """
        if not input:
            raise ValueError("OpenAI embeddings input cannot be empty.")

        params: dict[str, Any] = {"model": self.model_name, "input": input}
        if self.dimensions is not None and "text-embedding-3" in self.model_name:
            params["dimensions"] = self.dimensions

        response = self.client.embeddings.create(**params)
        return [np.array(item.embedding, dtype=np.float32) for item in response.data]
