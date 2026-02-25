"""Type definitions for Google embedding providers."""

from typing import Annotated, Literal

from typing_extensions import Required, TypedDict


class GenerativeAiProviderConfig(TypedDict, total=False):
    """Configuration for Google Generative AI provider.

    Attributes:
        api_key: Google API key for authentication.
        model_name: Embedding model name.
        task_type: Task type for embeddings. Default is "RETRIEVAL_DOCUMENT".
    """

    api_key: str
    model_name: Annotated[
        Literal[
            "gemini-embedding-001",
            "text-embedding-005",
            "text-multilingual-embedding-002",
        ],
        "gemini-embedding-001",
    ]
    task_type: Annotated[str, "RETRIEVAL_DOCUMENT"]


class GenerativeAiProviderSpec(TypedDict):
    """Google Generative AI provider specification."""

    provider: Literal["google-generativeai"]
    config: GenerativeAiProviderConfig


class VertexAIProviderConfig(TypedDict, total=False):
    """Configuration for Vertex AI provider with dual SDK support.

    Supports both legacy models (textembedding-gecko*) using the deprecated
    vertexai.language_models SDK and new models using google-genai SDK.

    Attributes:
        api_key: Google API key (optional if using project_id with ADC). Only for new SDK models.
        model_name: Embedding model name (default: "textembedding-gecko").
            Legacy models: textembedding-gecko, textembedding-gecko@001, etc.
            New models: gemini-embedding-001, text-embedding-005, text-multilingual-embedding-002
        project_id: GCP project ID (required for Vertex AI backend and legacy models).
        location: GCP region/location (default: "us-central1").
        region: Deprecated alias for location (kept for backwards compatibility).
        task_type: Task type for embeddings (default: "RETRIEVAL_DOCUMENT"). Only for new SDK models.
        output_dimensionality: Output embedding dimension (optional). Only for new SDK models.
    """

    api_key: str
    model_name: Annotated[
        Literal[
            # Legacy models (deprecated vertexai.language_models SDK)
            "textembedding-gecko",
            "textembedding-gecko@001",
            "textembedding-gecko@002",
            "textembedding-gecko@003",
            "textembedding-gecko@latest",
            "textembedding-gecko-multilingual",
            "textembedding-gecko-multilingual@001",
            "textembedding-gecko-multilingual@latest",
            # New models (google-genai SDK)
            "gemini-embedding-001",
            "text-embedding-005",
            "text-multilingual-embedding-002",
        ],
        "textembedding-gecko",
    ]
    project_id: str
    location: Annotated[str, "us-central1"]
    region: Annotated[str, "us-central1"]  # Deprecated alias for location
    task_type: Annotated[str, "RETRIEVAL_DOCUMENT"]
    output_dimensionality: int


class VertexAIProviderSpec(TypedDict, total=False):
    """Vertex AI provider specification."""

    provider: Required[Literal["google-vertex"]]
    config: VertexAIProviderConfig
