"""Google Vertex AI embeddings provider.

This module supports both the new google-genai SDK and the deprecated
vertexai.language_models module for backwards compatibility.

The SDK is automatically selected based on the model name:
- Legacy models (textembedding-gecko*) use vertexai.language_models (deprecated)
- New models (gemini-embedding-*, text-embedding-*) use google-genai

Migration guide: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/deprecations/genai-vertexai-sdk
"""

from __future__ import annotations

from pydantic import AliasChoices, Field

from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider
from crewai.rag.embeddings.providers.google.genai_vertex_embedding import (
    GoogleGenAIVertexEmbeddingFunction,
)


class VertexAIProvider(BaseEmbeddingsProvider[GoogleGenAIVertexEmbeddingFunction]):
    """Google Vertex AI embeddings provider with dual SDK support.

    Supports both legacy models (textembedding-gecko*) using the deprecated
    vertexai.language_models SDK and new models (gemini-embedding-*, text-embedding-*)
    using the google-genai SDK.

    The SDK is automatically selected based on the model name. Legacy models will
    emit a deprecation warning.

    Authentication modes:
    1. Vertex AI backend: Set project_id and location/region (uses Application Default Credentials)
    2. API key: Set api_key for direct API access (new SDK models only)

    Example:
        # Legacy model (backwards compatible, will emit deprecation warning)
        provider = VertexAIProvider(
            project_id="my-project",
            region="us-central1",  # or location="us-central1"
            model_name="textembedding-gecko"
        )

        # New model with Vertex AI backend
        provider = VertexAIProvider(
            project_id="my-project",
            location="us-central1",
            model_name="gemini-embedding-001"
        )

        # New model with API key
        provider = VertexAIProvider(
            api_key="your-api-key",
            model_name="gemini-embedding-001"
        )
    """

    embedding_callable: type[GoogleGenAIVertexEmbeddingFunction] = Field(
        default=GoogleGenAIVertexEmbeddingFunction,
        description="Google Vertex AI embedding function class",
    )
    model_name: str = Field(
        default="textembedding-gecko",
        description=(
            "Model name to use for embeddings. Legacy models (textembedding-gecko*) "
            "use the deprecated SDK. New models (gemini-embedding-001, text-embedding-005) "
            "use the google-genai SDK."
        ),
        validation_alias=AliasChoices(
            "EMBEDDINGS_GOOGLE_VERTEX_MODEL_NAME",
            "GOOGLE_VERTEX_MODEL_NAME",
            "model",
        ),
    )
    api_key: str | None = Field(
        default=None,
        description="Google API key (optional if using project_id with Application Default Credentials)",
        validation_alias=AliasChoices(
            "EMBEDDINGS_GOOGLE_CLOUD_API_KEY",
            "GOOGLE_CLOUD_API_KEY",
            "GOOGLE_API_KEY",
        ),
    )
    project_id: str | None = Field(
        default=None,
        description="GCP project ID (required for Vertex AI backend and legacy models)",
        validation_alias=AliasChoices(
            "EMBEDDINGS_GOOGLE_CLOUD_PROJECT",
            "GOOGLE_CLOUD_PROJECT",
        ),
    )
    location: str = Field(
        default="us-central1",
        description="GCP region/location",
        validation_alias=AliasChoices(
            "EMBEDDINGS_GOOGLE_CLOUD_LOCATION",
            "EMBEDDINGS_GOOGLE_CLOUD_REGION",
            "GOOGLE_CLOUD_LOCATION",
            "GOOGLE_CLOUD_REGION",
        ),
    )
    region: str | None = Field(
        default=None,
        description="Deprecated: Use 'location' instead. GCP region (kept for backwards compatibility)",
        validation_alias=AliasChoices(
            "EMBEDDINGS_GOOGLE_VERTEX_REGION",
            "GOOGLE_VERTEX_REGION",
        ),
    )
    task_type: str = Field(
        default="RETRIEVAL_DOCUMENT",
        description="Task type for embeddings (e.g., RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY). Only used with new SDK models.",
        validation_alias=AliasChoices(
            "EMBEDDINGS_GOOGLE_VERTEX_TASK_TYPE",
            "GOOGLE_VERTEX_TASK_TYPE",
        ),
    )
    output_dimensionality: int | None = Field(
        default=None,
        description="Output embedding dimensionality (optional). Only used with new SDK models.",
        validation_alias=AliasChoices(
            "EMBEDDINGS_GOOGLE_VERTEX_OUTPUT_DIMENSIONALITY",
            "GOOGLE_VERTEX_OUTPUT_DIMENSIONALITY",
        ),
    )
