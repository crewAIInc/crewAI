"""Type definitions for Ollama embedding providers."""

from typing import Annotated, Literal, Required, TypedDict


class OllamaProviderConfig(TypedDict, total=False):
    """Configuration for Ollama provider."""

    url: Annotated[str, "http://localhost:11434/api/embeddings"]
    model_name: Required[str]


class OllamaProviderSpec(TypedDict):
    """Ollama provider specification."""

    provider: Literal["ollama"]
    config: OllamaProviderConfig
