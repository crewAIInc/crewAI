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
    """Configuration for Vertex AI provider."""

    api_key: str
    model_name: Annotated[str, "textembedding-gecko"]
    project_id: Annotated[str, "cloud-large-language-models"]
    region: Annotated[str, "us-central1"]


class VertexAIProviderSpec(TypedDict, total=False):
    """Vertex AI provider specification."""

    provider: Required[Literal["google-vertex"]]
    config: VertexAIProviderConfig
