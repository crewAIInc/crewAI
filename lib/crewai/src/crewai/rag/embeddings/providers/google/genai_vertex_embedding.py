"""Google Vertex AI embedding function implementation.

This module supports both the new google-genai SDK and the deprecated
vertexai.language_models module for backwards compatibility.

The deprecated vertexai.language_models module will be removed after June 24, 2026.
Migration guide: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/deprecations/genai-vertexai-sdk
"""

from typing import Any, ClassVar, cast
import warnings

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from typing_extensions import Unpack

from crewai.rag.embeddings.providers.google.types import VertexAIProviderConfig


class GoogleGenAIVertexEmbeddingFunction(EmbeddingFunction[Documents]):
    """Embedding function for Google Vertex AI with dual SDK support.

    This class supports both:
    - Legacy models (textembedding-gecko*) using the deprecated vertexai.language_models SDK
    - New models (gemini-embedding-*, text-embedding-*) using the google-genai SDK

    The SDK is automatically selected based on the model name. Legacy models will
    emit a deprecation warning.

    Supports two authentication modes:
    1. Vertex AI backend: Set project_id and location/region (uses Application Default Credentials)
    2. API key: Set api_key for direct API access

    Example:
        # Using legacy model (will emit deprecation warning)
        embedder = GoogleGenAIVertexEmbeddingFunction(
            project_id="my-project",
            region="us-central1",
            model_name="textembedding-gecko"
        )

        # Using new model with google-genai SDK
        embedder = GoogleGenAIVertexEmbeddingFunction(
            project_id="my-project",
            location="us-central1",
            model_name="gemini-embedding-001"
        )

        # Using API key (new SDK only)
        embedder = GoogleGenAIVertexEmbeddingFunction(
            api_key="your-api-key",
            model_name="gemini-embedding-001"
        )
    """

    # Models that use the legacy vertexai.language_models SDK
    LEGACY_MODELS: ClassVar[set[str]] = {
        "textembedding-gecko",
        "textembedding-gecko@001",
        "textembedding-gecko@002",
        "textembedding-gecko@003",
        "textembedding-gecko@latest",
        "textembedding-gecko-multilingual",
        "textembedding-gecko-multilingual@001",
        "textembedding-gecko-multilingual@latest",
    }

    # Models that use the new google-genai SDK
    GENAI_MODELS: ClassVar[set[str]] = {
        "gemini-embedding-001",
        "text-embedding-005",
        "text-multilingual-embedding-002",
    }

    def __init__(self, **kwargs: Unpack[VertexAIProviderConfig]) -> None:
        """Initialize Google Vertex AI embedding function.

        Args:
            **kwargs: Configuration parameters including:
                - model_name: Model to use for embeddings (default: "textembedding-gecko")
                - api_key: Optional API key for authentication (new SDK only)
                - project_id: GCP project ID (for Vertex AI backend)
                - location: GCP region (default: "us-central1")
                - region: Deprecated alias for location
                - task_type: Task type for embeddings (default: "RETRIEVAL_DOCUMENT", new SDK only)
                - output_dimensionality: Optional output embedding dimension (new SDK only)
        """
        # Handle deprecated 'region' parameter (only if it has a value)
        region_value = kwargs.pop("region", None)  # type: ignore[typeddict-item]
        if region_value is not None:
            warnings.warn(
                "The 'region' parameter is deprecated, use 'location' instead. "
                "See: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/deprecations/genai-vertexai-sdk",
                DeprecationWarning,
                stacklevel=2,
            )
            if "location" not in kwargs or kwargs.get("location") is None:
                kwargs["location"] = region_value  # type: ignore[typeddict-unknown-key]

        self._config = kwargs
        self._model_name = str(kwargs.get("model_name", "textembedding-gecko"))
        self._use_legacy = self._is_legacy_model(self._model_name)

        if self._use_legacy:
            self._init_legacy_client(**kwargs)
        else:
            self._init_genai_client(**kwargs)

    def _is_legacy_model(self, model_name: str) -> bool:
        """Check if the model uses the legacy SDK."""
        return model_name in self.LEGACY_MODELS or model_name.startswith(
            "textembedding-gecko"
        )

    def _init_legacy_client(self, **kwargs: Any) -> None:
        """Initialize using the deprecated vertexai.language_models SDK."""
        warnings.warn(
            f"Model '{self._model_name}' uses the deprecated vertexai.language_models SDK "
            "which will be removed after June 24, 2026. Consider migrating to newer models "
            "like 'gemini-embedding-001'. "
            "See: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/deprecations/genai-vertexai-sdk",
            DeprecationWarning,
            stacklevel=3,
        )

        try:
            import vertexai
            from vertexai.language_models import TextEmbeddingModel
        except ImportError as e:
            raise ImportError(
                "vertexai is required for legacy embedding models (textembedding-gecko*). "
                "Install it with: pip install google-cloud-aiplatform"
            ) from e

        project_id = kwargs.get("project_id")
        location = str(kwargs.get("location", "us-central1"))

        if not project_id:
            raise ValueError(
                "project_id is required for legacy models. "
                "For API key authentication, use newer models like 'gemini-embedding-001'."
            )

        vertexai.init(project=str(project_id), location=location)
        self._legacy_model = TextEmbeddingModel.from_pretrained(self._model_name)

    def _init_genai_client(self, **kwargs: Any) -> None:
        """Initialize using the new google-genai SDK."""
        try:
            from google import genai
            from google.genai.types import EmbedContentConfig
        except ImportError as e:
            raise ImportError(
                "google-genai is required for Google Gen AI embeddings. "
                "Install it with: uv add 'crewai[google-genai]'"
            ) from e

        self._genai = genai
        self._EmbedContentConfig = EmbedContentConfig
        self._task_type = kwargs.get("task_type", "RETRIEVAL_DOCUMENT")
        self._output_dimensionality = kwargs.get("output_dimensionality")

        # Initialize client based on authentication mode
        api_key = kwargs.get("api_key")
        project_id = kwargs.get("project_id")
        location: str = str(kwargs.get("location", "us-central1"))

        if api_key:
            self._client = genai.Client(api_key=api_key)
        elif project_id:
            self._client = genai.Client(
                vertexai=True,
                project=str(project_id),
                location=location,
            )
        else:
            raise ValueError(
                "Either 'api_key' (for API key authentication) or 'project_id' "
                "(for Vertex AI backend with ADC) must be provided."
            )

    @staticmethod
    def name() -> str:
        """Return the name of the embedding function for ChromaDB compatibility."""
        return "google-vertex"

    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings for input documents.

        Args:
            input: List of documents to embed.

        Returns:
            List of embedding vectors.
        """
        if isinstance(input, str):
            input = [input]

        if self._use_legacy:
            return self._call_legacy(input)
        return self._call_genai(input)

    def _call_legacy(self, input: list[str]) -> Embeddings:
        """Generate embeddings using the legacy SDK."""
        import numpy as np

        embeddings_list = []
        for text in input:
            embedding_result = self._legacy_model.get_embeddings([text])
            embeddings_list.append(
                np.array(embedding_result[0].values, dtype=np.float32)
            )

        return cast(Embeddings, embeddings_list)

    def _call_genai(self, input: list[str]) -> Embeddings:
        """Generate embeddings using the new google-genai SDK."""
        # Build config for embed_content
        config_kwargs: dict[str, Any] = {
            "task_type": self._task_type,
        }
        if self._output_dimensionality is not None:
            config_kwargs["output_dimensionality"] = self._output_dimensionality

        config = self._EmbedContentConfig(**config_kwargs)

        # Call the embedding API
        response = self._client.models.embed_content(
            model=self._model_name,
            contents=input,  # type: ignore[arg-type]
            config=config,
        )

        # Extract embeddings from response
        if response.embeddings is None:
            raise ValueError("No embeddings returned from the API")
        embeddings = [emb.values for emb in response.embeddings]
        return cast(Embeddings, embeddings)
