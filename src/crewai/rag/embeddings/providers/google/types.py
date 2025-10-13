"""Type definitions for Google embedding providers."""

from typing import Annotated, Literal

from typing_extensions import Required, TypedDict


class GenerativeAiProviderConfig(TypedDict, total=False):
    """Configuration for Google Generative AI provider."""

    api_key: str
    model_name: Annotated[str, "models/embedding-001"]
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
